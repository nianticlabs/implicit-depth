import os
import sys

sys.path.append("/".join(sys.path[0].split("/")[:-1]))
from utils.metrics_utils import ResultsAverager

scores_path = "/mnt/res_nas/mohameds/simple_recon_output/deltas/scannet/default/iou_scores/all_frame_avg_metrics_test.json"

depths_for_printing = [1.5 + x * 0.5 for x in range(8)]
single_iou = True
all_frame_metrics = ResultsAverager("scores", f"frame metrics")
all_frame_metrics.from_json(scores_path)
thresholder = None

all_frame_metrics.pretty_print_metric_table(
    print_running_metrics=False, depths=depths_for_printing, single_iou=single_iou
)
all_frame_metrics.pretty_print_metric_table(
    print_running_metrics=False,
    metric_name="iou_neg",
    depths=depths_for_printing,
    single_iou=single_iou,
)
all_frame_metrics.pretty_print_metric_table(
    print_running_metrics=False,
    metric_name="iou_pos",
    depths=depths_for_printing,
    single_iou=single_iou,
)

all_frame_metrics.pretty_print_metric_table(
    print_running_metrics=False,
    metric_name="surface_iou",
    depths=depths_for_printing,
    single_iou=single_iou,
)
all_frame_metrics.pretty_print_metric_table(
    print_running_metrics=False,
    metric_name="surface_iou_neg",
    depths=depths_for_printing,
    single_iou=single_iou,
)
all_frame_metrics.pretty_print_metric_table(
    print_running_metrics=False,
    metric_name="surface_iou_pos",
    depths=depths_for_printing,
    single_iou=single_iou,
)

all_frame_metrics.pretty_print_metric_table(
    print_running_metrics=False,
    metric_name="boundary_iou",
    depths=depths_for_printing,
    single_iou=single_iou,
)
all_frame_metrics.pretty_print_metric_table(
    print_running_metrics=False,
    metric_name="boundary_iou_neg",
    depths=depths_for_printing,
    single_iou=single_iou,
)
all_frame_metrics.pretty_print_metric_table(
    print_running_metrics=False,
    metric_name="boundary_iou_pos",
    depths=depths_for_printing,
    single_iou=single_iou,
)
