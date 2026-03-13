import os
import csv
import glob
import statistics

def aggregate_hardware_profiles(input_dir="results/profiling", output_csv="results/profiling/hardware_summary.csv"):
    profile_files = glob.glob(os.path.join(input_dir, "*_profile_results.csv"))
    
    if not profile_files:
        print(f"No profiling results found in {input_dir}")
        return
        
    summary_data = []
    
    for file_path in profile_files:
        model_name = os.path.basename(file_path).replace("_profile_results.csv", "")
        vram_list = []
        time_list = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vram_list.append(float(row.get("peak_vram_gb", 0)))
                time_list.append(float(row.get("inference_time_sec", 0)))
                
        if not vram_list:
            continue
            
        avg_vram = statistics.mean(vram_list)
        peak_vram = max(vram_list)
        avg_time = statistics.mean(time_list)
        
        images_per_min = 60.0 / avg_time if avg_time > 0 else 0
        
        summary_data.append({
            "model": model_name,
            "images_profiled": len(time_list),
            "avg_peak_vram_gb": round(avg_vram, 3),
            "max_peak_vram_gb": round(peak_vram, 3),
            "avg_inference_time_sec": round(avg_time, 2),
            "images_per_minute": round(images_per_min, 2)
        })
        
    summary_data.sort(key=lambda x: x["images_per_minute"], reverse=True)
    
    # Save to CSV
    keys = summary_data[0].keys()
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_data)
        
    # Print a nice table
    print("=" * 100)
    print(f"{'Model':<15} | {'Profiled':<10} | {'Avg Peak VRAM':<15} | {'Max Peak VRAM':<15} | {'Avg Time (s)':<15} | {'Images/Min':<15}")
    print("-" * 100)
    for row in summary_data:
        print(f"{row['model']:<15} | {row['images_profiled']:<10} | {row['avg_peak_vram_gb']:<12} GB | {row['max_peak_vram_gb']:<12} GB | {row['avg_inference_time_sec']:<12} s | {row['images_per_minute']:<15}")
    print("=" * 100)
    print(f"Summary saved to {output_csv}")

if __name__ == "__main__":
    aggregate_hardware_profiles()
