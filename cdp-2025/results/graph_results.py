import csv
import matplotlib.pyplot as plt

RUNTIME_BENCHMARKS = {
    "benchmark_runtime_python_bihashmap.csv": "Python (bihashmap)",
    "benchmark_runtime_python_bimap_tuple.csv": "Python (bimap set tuple)",
    "benchmark_runtime_python_optimized_dict.csv": "Python (optimized dict)",
    "benchmark_runtime_rust_release.csv": "Rust (release)",
    "benchmark_runtime_rust_debug.csv": "Rust (debug)",
}

MEMORY_BENCHMARKS = {
    "benchmark_memory_python_bihashmap.csv": "Python (bihashmap)",
    "benchmark_memory_python_bimap_tuple.csv": "Python (bimap set tuple)",
    "benchmark_memory_python_optimized_dict.csv": "Python (optimized dict)",
}


def plot_benchmarks(benchmarks: dict[str, str], title: str, xlabel: str, ylabel: str, output_file: str, use_log_scale: bool = False) -> None:
    plt.figure(figsize=(10, 6))

    for file_name, label in benchmarks.items():
        input_size = []
        measured_value = []

        with open(f"results/{file_name}", "r") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            for row in csvreader:
                input_size.append(int(row[0]))
                measured_value.append(float(row[1]))

        if use_log_scale:
            plt.loglog(input_size, measured_value, marker="o", label=label)
        else:
            plt.plot(input_size, measured_value, marker="o", label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if use_log_scale:
        plt.savefig(f"results/loglog_{output_file}")
    else:
        plt.savefig(f"results/{output_file}")
    plt.close()


if __name__ == "__main__":
    plot_benchmarks(
        RUNTIME_BENCHMARKS,
        "Runtime Benchmark Results",
        "Input Size (Number of Workshops/Attendees)",
        "Time (seconds)",
        "runtime_benchmarks.png",
    )

    plot_benchmarks(
        MEMORY_BENCHMARKS,
        "Memory Benchmark Results",
        "Input Size (Number of Workshops/Attendees)",
        "Memory Usage (bytes)",
        "memory_benchmarks.png",
    )

    plot_benchmarks(
        RUNTIME_BENCHMARKS,
        "Runtime Benchmark Results",
        "Input Size (Number of Workshops/Attendees)",
        "Time (seconds)",
        "runtime_benchmarks.png",
        True,
    )

    plot_benchmarks(
        MEMORY_BENCHMARKS,
        "Memory Benchmark Results",
        "Input Size (Number of Workshops/Attendees)",
        "Memory Usage (bytes)",
        "memory_benchmarks.png",
        True,
    )
