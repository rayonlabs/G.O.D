import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from scipy.ndimage import uniform_filter1d
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

load_dotenv('.vali.env')

plt.style.use('dark_background')

def fetch_benchmark_data():
    validator_host = os.getenv('VALIDATOR_HOST', '185.141.218.75')
    validator_port = os.getenv('VALIDATOR_PORT', '8010')
    base_url = f"http://{validator_host}:{validator_port}"
    url = f"{base_url}/v1/benchmarks/timeline"
    
    api_key = os.getenv('FRONTEND_API_KEY')
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def moving_average_with_confidence(data, window=3):
    clean_data = [x for x in data if x is not None]
    
    if len(clean_data) < window:
        return clean_data, clean_data, clean_data
    
    ma = uniform_filter1d(clean_data, size=window, mode='nearest')
    
    rolling_std = pd.Series(clean_data).rolling(window=window, center=True, min_periods=1).std()
    rolling_std = rolling_std.fillna(rolling_std.mean()).values
    
    upper_band = ma + rolling_std * 0.5
    lower_band = ma - rolling_std * 0.5
    
    return ma, upper_band, lower_band

def plot_task_group(tasks, task_type, ax):
    colors = ['#00D9FF', '#FF00FF', '#00FF00', '#FFD700', '#FF6B6B', '#4ECDC4', '#95E77E']
    
    for idx, task in enumerate(tasks):
        benchmarks = task['benchmarks']
        test_losses = []
        
        for benchmark in benchmarks:
            if benchmark['test_loss'] is not None:
                test_losses.append(benchmark['test_loss'])
        
        if not test_losses:
            continue
            
        x = list(range(1, len(test_losses) + 1))
        
        ma, upper, lower = moving_average_with_confidence(test_losses, window=3)
        x_ma = list(range(1, len(ma) + 1))
        
        color = colors[idx % len(colors)]
        
        ax.fill_between(x_ma, lower, upper, alpha=0.15, color=color)
        
        model_name = task['model_id'].split('/')[-1][:20]
        ax.plot(x_ma, ma, color=color, linewidth=2.5, alpha=0.9, label=model_name)
    
    ax.set_xlabel('Tournament', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    ax.set_title(f'{task_type.replace("Task", "")} Performance', fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    if x_ma:
        ax.set_xticks(range(1, max(x_ma) + 1))
    
    if len(tasks) > 1:
        ax.legend(frameon=False, loc='best', fontsize=9, ncol=2)

def main():
    print("Fetching benchmark data from localhost:8010...")
    data = fetch_benchmark_data()
    
    if not data or 'timelines' not in data:
        print("No data received or invalid format")
        return
    
    timelines = data['timelines']
    
    image_tasks = [t for t in timelines if t['task_type'] == 'ImageTask']
    text_tasks = [t for t in timelines if t['task_type'] in ['InstructTextTask', 'DpoTask']]
    
    fig_width = 16
    fig_height = 6
    
    if image_tasks and text_tasks:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        fig.patch.set_facecolor('#0d1117')
        
        plot_task_group(image_tasks, 'Image Tasks', ax1)
        plot_task_group(text_tasks, 'Text Tasks', ax2)
        
    elif image_tasks:
        fig, ax = plt.subplots(1, 1, figsize=(fig_width//2, fig_height))
        fig.patch.set_facecolor('#0d1117')
        plot_task_group(image_tasks, 'Image Tasks', ax)
        
    elif text_tasks:
        fig, ax = plt.subplots(1, 1, figsize=(fig_width//2, fig_height))
        fig.patch.set_facecolor('#0d1117')
        plot_task_group(text_tasks, 'Text Tasks', ax)
    
    else:
        print("No valid tasks with test losses found")
        return
    
    plt.tight_layout()
    
    os.makedirs('performance_charts/figures', exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'performance_charts/figures/benchmark_performance_{date_str}.png'
    plt.savefig(output_path, dpi=150, facecolor='#0d1117', edgecolor='none', bbox_inches='tight')
    print(f"\nChart saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()