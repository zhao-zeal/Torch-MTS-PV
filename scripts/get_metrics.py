import os
import re
import sys

sys.path.append("..")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(BASE_DIR, "logs")
METRIC_COLUMNS = ("MAE", "RMSE", "MAPE", "ACC_MAE", "ACC_RMSE")
METRIC_LINE_RE = re.compile(
    r"^(?P<scope>All Steps \(1-(?P<horizon>\d+)\)|Step (?P<step>\d+))\s+"
    r"MAE = (?P<MAE>[-+]?\d*\.?\d+),\s+"
    r"RMSE = (?P<RMSE>[-+]?\d*\.?\d+),\s+"
    r"(?:(?:MAPE = (?P<MAPE>[-+]?\d*\.?\d+))|"
    r"(?:ACC_MAE = (?P<ACC_MAE>[-+]?\d*\.?\d+),\s+"
    r"ACC_RMSE = (?P<ACC_RMSE>[-+]?\d*\.?\d+)))"
)


def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()


def get_metrics_log(log: str):
    with open(log, "r") as f:
        lines = f.readlines()

    metrics = []
    for line in lines:
        match = METRIC_LINE_RE.search(line)
        if not match:
            continue

        step = int(match.group("step") or match.group("horizon"))
        row = {"Step": step}
        for metric in METRIC_COLUMNS:
            value = match.group(metric)
            if value is not None:
                row[metric] = float(value)

        metrics.append(row)

    return metrics


def print_model_metrics(model: str, dataset=None, file=None):
    model_logs = os.path.join(log_path, model)
    for log in sorted(os.listdir(model_logs)):
        if dataset:
            if model not in log or dataset.upper() not in log:
                continue

        print_log(log, log=file)
        for line in get_metrics_log(os.path.join(model_logs, log)):
            values = [line["Step"]] + [
                line[metric] for metric in METRIC_COLUMNS if metric in line
            ]
            for value in values:
                if value % 1 == 0:
                    print_log(int(value), end="\t", log=file)
                else:
                    print_log("%.4f" % value, end="\t\t", log=file)
            print_log(log=file)
        print_log(log=file)


def print_model_metrics_csv(models, datasets, file=None):
    print_log("Dataset,Model,Log,Step," + ",".join(METRIC_COLUMNS), log=file)

    for dataset in datasets:
        for model in models:
            model_logs = os.path.join(log_path, model)
            for log in sorted(os.listdir(model_logs)):
                if dataset:
                    if model not in log or dataset.upper() not in log:
                        continue

                for line in get_metrics_log(os.path.join(model_logs, log)):
                    values = [
                        f"{line[metric]:.4f}" if metric in line else ""
                        for metric in METRIC_COLUMNS
                    ]
                    print_log(
                        f"{dataset.upper()},{model},{log},{line['Step']},"
                        + ",".join(values),
                        log=file,
                    )
                print_log(log=file)


def print_model_metrics_csv_long(models, datasets, file=None):
    rows = []
    headers = ["Dataset", "Model"]
    for dataset in datasets:
        for model in models:
            model_logs = os.path.join(log_path, model)
            row = {"Dataset": dataset.upper(), "Model": model}
            for log in sorted(os.listdir(model_logs)):
                if dataset:
                    if model not in log or dataset.upper() not in log:
                        continue

                for line in get_metrics_log(os.path.join(model_logs, log)):
                    for metric in METRIC_COLUMNS:
                        if metric not in line:
                            continue
                        header = f"{metric}_{line['Step']}"
                        if header not in headers:
                            headers.append(header)
                        row[header] = f"{line[metric]:.4f}"
            if len(row) > 2:
                rows.append(row)

    print_log(",".join(headers), log=file)
    for row in rows:
        print_log(",".join(row.get(header, "") for header in headers), log=file)


if __name__ == "__main__":
    models = [
        "DLinear",
    ]
    datasets = ["SKIPPD"]

    # for dataset in datasets:
    #     for model in models:
    #         print_model_metrics(model, dataset)

    file = open(os.path.join(log_path, "results.csv"), "a")
    file.seek(0)
    file.truncate()
    print_model_metrics_csv(models, datasets, file=file)
    
    file = open(os.path.join(log_path, "results_long.csv"), "a")
    file.seek(0)
    file.truncate()
    print_model_metrics_csv_long(models, datasets, file=file)
