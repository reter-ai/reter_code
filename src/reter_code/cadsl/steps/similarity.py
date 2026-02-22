"""
CADSL Similarity Steps.

Contains step classes for similarity computation in CADSL pipelines:
- SetSimilarityStep: Compute set similarity (jaccard, dice, overlap, cosine)
- StringMatchStep: Detect string pattern matches (common_affix, contains, levenshtein)
"""

from typing import Any, Dict, List, Optional


class SetSimilarityStep:
    """
    Compute set similarity between two columns.

    Syntax: set_similarity { left: col1, right: col2, type: jaccard, output: similarity }

    Types:
    - jaccard: |intersection| / |union|
    - dice: 2 * |intersection| / (|A| + |B|)
    - overlap: |intersection| / min(|A|, |B|)
    - cosine: |intersection| / sqrt(|A| * |B|)

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, left_col, right_col, sim_type="jaccard", output="similarity",
                 intersection_output=None, union_output=None):
        self.left_col = left_col
        self.right_col = right_col
        self.sim_type = sim_type
        self.output = output
        self.intersection_output = intersection_output
        self.union_output = union_output

    def execute(self, data, ctx=None):
        """Calculate set similarity."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            if not data:
                return pipeline_ok([])

            result = []
            for row in data:
                new_row = dict(row)
                left = row.get(self.left_col) or []
                right = row.get(self.right_col) or []

                # Convert to sets
                left_set = set(left) if isinstance(left, (list, tuple, set)) else {left}
                right_set = set(right) if isinstance(right, (list, tuple, set)) else {right}

                intersection = left_set & right_set
                union = left_set | right_set

                # Calculate similarity
                if self.sim_type == "jaccard":
                    similarity = len(intersection) / len(union) if union else 0
                elif self.sim_type == "dice":
                    total = len(left_set) + len(right_set)
                    similarity = 2 * len(intersection) / total if total else 0
                elif self.sim_type == "overlap":
                    min_size = min(len(left_set), len(right_set))
                    similarity = len(intersection) / min_size if min_size else 0
                elif self.sim_type == "cosine":
                    denom = (len(left_set) * len(right_set)) ** 0.5
                    similarity = len(intersection) / denom if denom else 0
                else:
                    similarity = len(intersection) / len(union) if union else 0

                new_row[self.output] = round(similarity, 4)

                if self.intersection_output:
                    new_row[self.intersection_output] = list(intersection)
                if self.union_output:
                    new_row[self.union_output] = list(union)

                result.append(new_row)

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("set_similarity", f"Set similarity failed: {e}", e)


class StringMatchStep:
    """
    Detect string pattern matches between two columns.

    Syntax: string_match { left: col1, right: col2, type: common_affix, min_length: 3, output: has_match }

    Types:
    - common_affix: Check for common prefix OR suffix
    - common_prefix: Check for common prefix only
    - common_suffix: Check for common suffix only
    - levenshtein: Calculate edit distance (requires output_distance)
    - contains: Check if one contains the other

    ::: This is-in-layer Domain-Specific-Language-Layer.
    ::: This is a step.
    ::: This is-in-process Main-Process.
    ::: This is stateless.
    """

    def __init__(self, left_col, right_col, match_type="common_affix", min_length=3,
                 output="has_match", match_output=None):
        self.left_col = left_col
        self.right_col = right_col
        self.match_type = match_type
        self.min_length = min_length
        self.output = output
        self.match_output = match_output

    def execute(self, data, ctx=None):
        """Execute string matching."""
        from reter_code.dsl.core import pipeline_ok, pipeline_err

        try:
            # Convert to list if Arrow table
            if hasattr(data, 'to_pylist'):
                data = data.to_pylist()

            if not data:
                return pipeline_ok([])

            result = []
            for row in data:
                new_row = dict(row)
                left = str(row.get(self.left_col, ""))
                right = str(row.get(self.right_col, ""))

                has_match = False
                match_value = None

                if self.match_type in ("common_affix", "common_prefix", "common_suffix"):
                    # Check prefix
                    if self.match_type in ("common_affix", "common_prefix"):
                        min_len = min(len(left), len(right))
                        for plen in range(self.min_length, min_len + 1):
                            if left[:plen] == right[:plen]:
                                # Found common prefix, check if different suffixes
                                s1, s2 = left[plen:], right[plen:]
                                if s1 and s2 and s1 != s2:
                                    has_match = True
                                    match_value = f"prefix:{left[:plen]}"
                                    break

                    # Check suffix
                    if not has_match and self.match_type in ("common_affix", "common_suffix"):
                        min_len = min(len(left), len(right))
                        for slen in range(self.min_length, min_len + 1):
                            if left[-slen:] == right[-slen:]:
                                # Found common suffix, check if different prefixes
                                p1, p2 = left[:-slen], right[:-slen]
                                if p1 and p2 and p1 != p2:
                                    has_match = True
                                    match_value = f"suffix:{left[-slen:]}"
                                    break

                elif self.match_type == "contains":
                    if left in right:
                        has_match = True
                        match_value = f"left_in_right:{left}"
                    elif right in left:
                        has_match = True
                        match_value = f"right_in_left:{right}"

                elif self.match_type == "levenshtein":
                    # Simple Levenshtein distance
                    distance = self._levenshtein(left, right)
                    has_match = distance <= self.min_length
                    match_value = distance

                new_row[self.output] = has_match
                if self.match_output and match_value is not None:
                    new_row[self.match_output] = match_value

                result.append(new_row)

            return pipeline_ok(result)
        except Exception as e:
            return pipeline_err("string_match", f"String match failed: {e}", e)

    def _levenshtein(self, s1, s2):
        """Calculate Levenshtein distance."""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]
