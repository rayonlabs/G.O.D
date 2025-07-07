"""
Tournament Structure Visualization
Creates high-quality tournament bracket and structure diagrams
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
import sys
import os

# Add the parent directory to sys.path to import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from validator.tournament.constants import (
        MAX_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND,
        EXPECTED_GROUP_SIZE,
        MIN_GROUP_SIZE,
        TEXT_TASKS_PER_GROUP,
        IMAGE_TASKS_PER_GROUP
    )
    from validator.core.constants import (
        TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100,
        TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100,
        TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100
    )
except ImportError:
    # Fallback constants if import fails
    MAX_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND = 16
    EXPECTED_GROUP_SIZE = 8
    MIN_GROUP_SIZE = 6
    TEXT_TASKS_PER_GROUP = 3
    IMAGE_TASKS_PER_GROUP = 1
    TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100 = 4.0
    TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100 = 12.0
    TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100 = 40.0


class TournamentStructureVisualizer:
    
    def __init__(self):
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'background': '#F5F5F5',
            'text': '#2C3E50',
            'group_stage': '#3498DB',
            'knockout': '#E74C3C',
            'boss': '#F39C12',
            'gpu_1x': '#95A5A6',
            'gpu_2x': '#3498DB',
            'gpu_4x': '#E74C3C',
            'gpu_8x': '#9B59B6'
        }
        
    def create_tournament_bracket(self, num_miners, tournament_type='TEXT'):
        """Create a tournament bracket visualization"""
        
        # Determine tournament structure
        is_group_stage = num_miners > MAX_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND
        
        if is_group_stage:
            return self._create_group_stage_visualization(num_miners, tournament_type)
        else:
            return self._create_knockout_visualization(num_miners, tournament_type)
    
    def _create_group_stage_visualization(self, num_miners, tournament_type):
        """Create group stage tournament visualization"""
        
        # Calculate group structure
        num_groups = math.ceil(num_miners / EXPECTED_GROUP_SIZE)
        miners_per_group = math.ceil(num_miners / num_groups)
        
        # Create subplot grid
        cols = min(4, num_groups)
        rows = math.ceil(num_groups / cols)
        
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[f"Group {i+1}" for i in range(num_groups)],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Add group visualizations
        for group_idx in range(num_groups):
            row = group_idx // cols + 1
            col = group_idx % cols + 1
            
            # Calculate miners in this group
            start_miner = group_idx * miners_per_group
            end_miner = min(start_miner + miners_per_group, num_miners)
            group_miners = end_miner - start_miner
            
            # Create group grid
            x_positions = []
            y_positions = []
            miner_names = []
            
            # Arrange miners in a circular pattern
            for i in range(group_miners):
                angle = 2 * math.pi * i / group_miners
                x = math.cos(angle)
                y = math.sin(angle)
                x_positions.append(x)
                y_positions.append(y)
                miner_names.append(f"Miner {start_miner + i + 1}")
            
            # Add center node for tasks
            x_positions.append(0)
            y_positions.append(0)
            tasks_per_group = TEXT_TASKS_PER_GROUP if tournament_type == 'TEXT' else IMAGE_TASKS_PER_GROUP
            miner_names.append(f"{tasks_per_group} Tasks")
            
            # Add scatter plot for this group
            fig.add_trace(
                go.Scatter(
                    x=x_positions[:-1],
                    y=y_positions[:-1],
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color=self.colors['group_stage'],
                        line=dict(width=2, color='white')
                    ),
                    text=miner_names[:-1],
                    textposition="middle center",
                    textfont=dict(size=8, color='white'),
                    name=f"Group {group_idx + 1}",
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add central tasks node
            fig.add_trace(
                go.Scatter(
                    x=[x_positions[-1]],
                    y=[y_positions[-1]],
                    mode='markers+text',
                    marker=dict(
                        size=30,
                        color=self.colors['accent'],
                        line=dict(width=2, color='white')
                    ),
                    text=[miner_names[-1]],
                    textposition="middle center",
                    textfont=dict(size=10, color='white', weight='bold'),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add connecting lines
            for i in range(group_miners):
                fig.add_trace(
                    go.Scatter(
                        x=[x_positions[i], x_positions[-1]],
                        y=[y_positions[i], y_positions[-1]],
                        mode='lines',
                        line=dict(color=self.colors['primary'], width=1),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Tournament Structure - Group Stage<br><sub>{num_miners} Miners, {num_groups} Groups, {tournament_type} Tournament</sub>",
                x=0.5,
                font=dict(size=20, color=self.colors['text'])
            ),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400 * rows,
            width=1200,
            font=dict(color=self.colors['text'])
        )
        
        # Update axes
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
        
        return fig
    
    def _create_knockout_visualization(self, num_miners, tournament_type):
        """Create knockout tournament bracket"""
        
        # Calculate bracket structure
        rounds = math.ceil(math.log2(num_miners))
        
        # Create positions for bracket
        fig = go.Figure()
        
        # Colors for different rounds
        round_colors = [self.colors['knockout'], self.colors['primary'], self.colors['secondary']]
        
        # Draw bracket structure
        for round_num in range(rounds):
            matches_in_round = 2 ** (rounds - round_num - 1)
            participants_in_round = matches_in_round * 2 if round_num == 0 else matches_in_round
            
            # Calculate positions
            x_pos = round_num
            y_spacing = 4.0 / matches_in_round if matches_in_round > 0 else 4.0
            
            for match_idx in range(matches_in_round):
                y_pos = (match_idx + 0.5) * y_spacing - 2.0
                
                # Add match node
                color = round_colors[round_num % len(round_colors)]
                if round_num == rounds - 1:  # Final round
                    color = self.colors['boss']
                    text = "Boss Round"
                    size = 40
                else:
                    text = f"R{round_num + 1}M{match_idx + 1}"
                    size = 30
                
                fig.add_trace(
                    go.Scatter(
                        x=[x_pos],
                        y=[y_pos],
                        mode='markers+text',
                        marker=dict(
                            size=size,
                            color=color,
                            line=dict(width=2, color='white')
                        ),
                        text=text,
                        textposition="middle center",
                        textfont=dict(size=10, color='white', weight='bold'),
                        showlegend=False
                    )
                )
                
                # Add connecting lines to next round
                if round_num < rounds - 1:
                    next_match_idx = match_idx // 2
                    next_y_pos = (next_match_idx + 0.5) * (4.0 / (matches_in_round // 2)) - 2.0
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[x_pos + 0.3, x_pos + 0.7],
                            y=[y_pos, next_y_pos],
                            mode='lines',
                            line=dict(color=self.colors['text'], width=2),
                            showlegend=False
                        )
                    )
        
        # Add initial participants
        for i in range(min(num_miners, 16)):
            y_pos = (i + 0.5) * (4.0 / min(num_miners, 16)) - 2.0
            fig.add_trace(
                go.Scatter(
                    x=[-0.5],
                    y=[y_pos],
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color=self.colors['group_stage'],
                        line=dict(width=2, color='white')
                    ),
                    text=f"M{i+1}",
                    textposition="middle center",
                    textfont=dict(size=8, color='white'),
                    showlegend=False
                )
            )
            
            # Connect to first round
            if i < 2 ** (rounds - 1):
                match_idx = i // 2
                match_y = (match_idx + 0.5) * (4.0 / (2 ** (rounds - 1))) - 2.0
                fig.add_trace(
                    go.Scatter(
                        x=[-0.2, 0.0],
                        y=[y_pos, match_y],
                        mode='lines',
                        line=dict(color=self.colors['text'], width=1),
                        showlegend=False
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Tournament Structure - Knockout Bracket<br><sub>{num_miners} Miners, {rounds} Rounds, {tournament_type} Tournament</sub>",
                x=0.5,
                font=dict(size=20, color=self.colors['text'])
            ),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            width=1200,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            font=dict(color=self.colors['text'])
        )
        
        return fig
    
    def create_gpu_allocation_chart(self, model_sizes_gb):
        """Create GPU allocation visualization based on model sizes"""
        
        gpu_configs = []
        colors = []
        
        for size in model_sizes_gb:
            if size < TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100:
                gpu_configs.append("1x H100")
                colors.append(self.colors['gpu_1x'])
            elif size < TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100:
                gpu_configs.append("2x H100")
                colors.append(self.colors['gpu_2x'])
            elif size < TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100:
                gpu_configs.append("4x H100")
                colors.append(self.colors['gpu_4x'])
            else:
                gpu_configs.append("8x H100")
                colors.append(self.colors['gpu_8x'])
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=[f"Model {i+1}" for i in range(len(model_sizes_gb))],
                y=model_sizes_gb,
                marker=dict(color=colors),
                text=gpu_configs,
                textposition='auto',
                textfont=dict(color='white', weight='bold')
            )
        )
        
        fig.update_layout(
            title="GPU Allocation by Model Size",
            xaxis_title="Models",
            yaxis_title="Model Size (GB)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color=self.colors['text'])
        )
        
        return fig


def generate_tournament_structure_charts(output_dir="output"):
    """Generate all tournament structure charts"""
    
    visualizer = TournamentStructureVisualizer()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate different tournament scenarios
    scenarios = [
        (8, 'TEXT', 'small_text_tournament'),
        (24, 'TEXT', 'large_text_tournament'),
        (12, 'IMAGE', 'image_tournament'),
        (32, 'TEXT', 'mega_text_tournament')
    ]
    
    for num_miners, tournament_type, filename in scenarios:
        fig = visualizer.create_tournament_bracket(num_miners, tournament_type)
        
        # Save as HTML
        fig.write_html(f"{output_dir}/{filename}.html")
        
        # Save as PNG
        fig.write_image(f"{output_dir}/{filename}.png", width=1200, height=800)
        
        print(f"Generated {filename} tournament structure visualization")
    
    # Generate GPU allocation chart
    model_sizes = [1.5, 7.0, 15.0, 45.0, 70.0]
    gpu_fig = visualizer.create_gpu_allocation_chart(model_sizes)
    gpu_fig.write_html(f"{output_dir}/gpu_allocation.html")
    gpu_fig.write_image(f"{output_dir}/gpu_allocation.png", width=1200, height=600)
    
    print("Generated GPU allocation visualization")


if __name__ == "__main__":
    generate_tournament_structure_charts()