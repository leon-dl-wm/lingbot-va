"""Evaluation result statistics script: compute RoboTwin per-task success rates from the
visualization video filenames saved by the evaluation client.

Role in the evaluation loop: after each episode, ``eval_polict_client_openpi.py`` saves a
video as ``<save_root>/stseed-*/visualization/<task_name>/<id>_<prompt>_<True|False>.mp4``,
where the trailing True/False directly marks the episode outcome; this script recursively
scans those mp4 filenames, counts successes/failures and computes the success rate per task,
and prints a table with the overall mean and per-class (CLASS 1/2/3) means.

Typical usage::

    python evaluation/robotwin/calc_stat.py results/stseed-10000/visualization [more roots...]
"""
from pathlib import Path

def compute_success_rates(root_dir: str, true_suffix="True.mp4", false_suffix="False.mp4"):
    """Scan all task subdirectories under the root directory, count success/failure videos
    by filename suffix, and compute success rates.

    Args:
        root_dir: Result root directory; each first-level subdirectory name corresponds to a task.
        true_suffix: Filename suffix of success videos (default "True.mp4").
        false_suffix: Filename suffix of failure videos (default "False.mp4").
    Returns:
        list of tuples ``(task name, success count, failure count, total, success rate)``;
        the rate is None when the total is 0.
    Raises:
        FileNotFoundError: If the root directory does not exist.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root dir not found: {root}")

    results = []
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        true_cnt = 0
        false_cnt = 0

        for mp4 in sub.rglob("*.mp4"):
            name = mp4.name
            if name.endswith(true_suffix):
                true_cnt += 1
            elif name.endswith(false_suffix):
                false_cnt += 1

        total = true_cnt + false_cnt
        rate = (true_cnt / total) if total > 0 else None
        results.append((sub.name, true_cnt, false_cnt, total, rate))

    return results


# 你的三类：task -> 1/2/3
# Task classification mapping: splits the 50 RoboTwin tasks into CLASS 1/2/3 by
# difficulty/type, used by print_table to report per-class mean success rates;
# tasks missing from this table are grouped into UNKNOWN and reported separately.
TASK_CLASS = {
    "adjust_bottle": 1,
    "beat_block_hammer": 1,
    "blocks_ranking_rgb": 3,
    "blocks_ranking_size": 3,
    "click_alarmclock": 1,
    "click_bell": 1,
    "dump_bin_bigbin": 1,
    "grab_roller": 1,
    "handover_block": 2,
    "handover_mic": 2,
    "hanging_mug": 2,
    "lift_pot": 1,
    "move_can_pot": 1,
    "move_pillbottle_pad": 1,
    "move_playingcard_away": 1,
    "move_stapler_pad": 1,
    "open_laptop": 1,
    "open_microwave": 1,
    "pick_diverse_bottles": 2,
    "pick_dual_bottles": 2,
    "place_a2b_left": 1,
    "place_a2b_right": 1,
    "place_bread_basket": 1,
    "place_bread_skillet": 2,
    "place_burger_fries": 2,
    "place_can_basket": 2,
    "place_cans_plasticbox": 2,
    "place_container_plate": 1,
    "place_dual_shoes": 2,
    "place_empty_cup": 1,
    "place_fan": 1,
    "place_mouse_pad": 1,
    "place_object_basket": 2,
    "place_object_scale": 1,
    "place_object_stand": 1,
    "place_phone_stand": 1,
    "place_shoe": 1,
    "press_stapler": 1,
    "put_bottles_dustbin": 3,
    "put_object_cabinet": 2,
    "rotate_qrcode": 1,
    "scan_object": 2,
    "shake_bottle_horizontally": 1,
    "shake_bottle": 1,
    "stack_blocks_three": 3,
    "stack_blocks_two": 2,
    "stack_bowls_three": 3,
    "stack_bowls_two": 2,
    "stamp_seal": 1,
    "turn_switch": 1,
}

def mean_rate_of(results_subset):
    """Compute the arithmetic mean of success rates over a subset of results; None (N/A)
    entries are ignored, and None is returned if the subset is all None.

    Args:
        results_subset: Subset of the tuples returned by compute_success_rates
            (the 5th element is the success rate).
    """
    rates = [r[4] for r in results_subset if r[4] is not None]
    return (sum(rates) / len(rates)) if rates else None


def print_table(results):
    """Print the statistics table sorted by success rate (descending), with extra rows for
    the overall mean, per-class (1/2/3) means, and the UNKNOWN mean.

    Args:
        results: List of tuples returned by compute_success_rates:
            ``(task name, success count, failure count, total, success rate)``.
    """
    # 按成功率排序：None(=N/A) 放最后，其余从高到低
    results = sorted(results, key=lambda r: (r[4] is None, -(r[4] or 0.0)))

    print(f"{'folder':30s} {'True':>6s} {'False':>6s} {'Total':>6s} {'SuccessRate':>12s} {'Class':>6s}")
    print("-" * 90)

    for folder, t, f, total, rate in results:
        rate_str = "N/A" if rate is None else f"{rate*100:9.2f}%"
        cls = TASK_CLASS.get(folder, None)
        cls_str = "N/A" if cls is None else str(cls)
        print(f"{folder:30s} {t:6d} {f:6d} {total:6d} {rate_str:>12s} {cls_str:>6s}")

    print("-" * 90)

    # overall mean
    overall_mean = mean_rate_of(results)
    overall_str = "N/A" if overall_mean is None else f"{overall_mean*100:9.2f}%"
    print(f"{'MEAN (ALL)':30s} {'':6s} {'':6s} {'':6s} {overall_str:>12s}")

    # per-class mean (1/2/3)
    for c in (1, 2, 3):
        subset = [r for r in results if TASK_CLASS.get(r[0]) == c]
        m = mean_rate_of(subset)
        m_str = "N/A" if m is None else f"{m*100:9.2f}%"
        print(f"{('MEAN (CLASS '+str(c)+')'):30s} {'':6s} {'':6s} {'':6s} {m_str:>12s}")

    # optional: tasks not in mapping
    unknown_subset = [r for r in results if r[0] not in TASK_CLASS]
    if unknown_subset:
        m = mean_rate_of(unknown_subset)
        m_str = "N/A" if m is None else f"{m*100:9.2f}%"
        print(f"{'MEAN (UNKNOWN)':30s} {'':6s} {'':6s} {'':6s} {m_str:>12s}")


# Command-line entry: accepts multiple result root directories, merges the statistics of
# all tasks, and prints a single combined table.
if __name__ == "__main__":
    import sys

    roots = sys.argv[1:]
    if not roots:
        raise SystemExit("Usage: python a.py <root_folder1> [<root_folder2> ...]")

    all_results = []
    for root_dir in roots:
        all_results.extend(compute_success_rates(root_dir))

    print_table(all_results)