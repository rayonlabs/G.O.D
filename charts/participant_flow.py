"""
Participant Flow Visualization
Creates high-quality flow diagrams showing the tournament participant journey
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ParticipantFlowVisualizer:
    
    def __init__(self):
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'background': '#F5F5F5',
            'text': '#2C3E50',
            'pending': '#95A5A6',
            'active': '#3498DB',
            'completed': '#2ECC71',
            'failed': '#E74C3C',
            'registration': '#9B59B6',
            'training': '#E67E22',
            'evaluation': '#F39C12',
            'advancement': '#27AE60'
        }
    
    def create_participant_lifecycle_flow(self):
        """Create a comprehensive participant lifecycle flow diagram"""
        
        # Define the flow stages
        stages = [
            {"name": "Registration", "color": self.colors['registration'], "x": 0, "y": 0},
            {"name": "Eligibility Check", "color": self.colors['pending'], "x": 1, "y": 0},
            {"name": "Tournament\nParticipation", "color": self.colors['active'], "x": 2, "y": 0},
            {"name": "Task Assignment", "color": self.colors['training'], "x": 3, "y": 0},
            {"name": "GPU Allocation", "color": self.colors['accent'], "x": 4, "y": 0},
            {"name": "Training", "color": self.colors['training'], "x": 5, "y": 0},
            {"name": "Evaluation", "color": self.colors['evaluation'], "x": 6, "y": 0},
            {"name": "Scoring", "color": self.colors['success'], "x": 7, "y": 0},
            {"name": "Advancement", "color": self.colors['advancement'], "x": 8, "y": 0}
        ]
        
        # Decision points
        decision_points = [
            {"name": "Eligible?", "x": 1, "y": -1, "success_x": 2, "fail_x": 1, "fail_y": -2},
            {"name": "Group vs\nKnockout?", "x": 2, "y": 1, "group_x": 3, "group_y": 1, "knockout_x": 3, "knockout_y": -1},
            {"name": "Training\nComplete?", "x": 5, "y": -1, "success_x": 6, "retry_x": 5, "retry_y": -2},
            {"name": "Advance?", "x": 8, "y": -1, "advance_x": 9, "advance_y": 0, "eliminate_x": 8, "eliminate_y": -2}
        ]
        
        fig = go.Figure()
        
        # Add main flow stages
        for i, stage in enumerate(stages):
            fig.add_trace(
                go.Scatter(
                    x=[stage["x"]],
                    y=[stage["y"]],
                    mode='markers+text',
                    marker=dict(
                        size=50,
                        color=stage["color"],
                        line=dict(width=3, color='white')
                    ),
                    text=stage["name"],
                    textposition="middle center",
                    textfont=dict(size=10, color='white', weight='bold'),
                    name=stage["name"],
                    showlegend=False
                )
            )
            
            # Add arrows between stages
            if i < len(stages) - 1:
                fig.add_annotation(
                    x=stage["x"] + 0.4,
                    y=stage["y"],
                    ax=stage["x"] + 0.6,
                    ay=stage["y"],
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=self.colors['text']
                )
        
        # Add decision points
        for decision in decision_points:
            fig.add_trace(
                go.Scatter(
                    x=[decision["x"]],
                    y=[decision["y"]],
                    mode='markers+text',
                    marker=dict(
                        size=40,
                        color=self.colors['secondary'],
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    text=decision["name"],
                    textposition="middle center",
                    textfont=dict(size=8, color='white', weight='bold'),
                    showlegend=False
                )
            )
        
        # Add specific decision flows
        # Eligibility check
        fig.add_annotation(x=1.3, y=-0.3, ax=2, ay=0, arrowhead=2, arrowcolor='green', text="✓ Pass")
        fig.add_annotation(x=1.3, y=-0.7, ax=1, ay=-2, arrowhead=2, arrowcolor='red', text="✗ Fail")
        
        # Group vs Knockout
        fig.add_annotation(x=2.3, y=0.7, ax=3, ay=1, arrowhead=2, arrowcolor='blue', text="16+ miners")
        fig.add_annotation(x=2.3, y=0.3, ax=3, ay=-1, arrowhead=2, arrowcolor='orange', text="<16 miners")
        
        # Training complete
        fig.add_annotation(x=5.3, y=-0.3, ax=6, ay=0, arrowhead=2, arrowcolor='green', text="✓ Complete")
        fig.add_annotation(x=5.3, y=-0.7, ax=5, ay=-2, arrowhead=2, arrowcolor='red', text="↻ Retry")
        
        # Advancement
        fig.add_annotation(x=8.3, y=-0.3, ax=9, ay=0, arrowhead=2, arrowcolor='green', text="✓ Advance")
        fig.add_annotation(x=8.3, y=-0.7, ax=8, ay=-2, arrowhead=2, arrowcolor='red', text="✗ Eliminate")
        
        # Add terminal states
        terminal_states = [
            {"name": "Rejected", "x": 1, "y": -2, "color": self.colors['failed']},
            {"name": "Group Stage", "x": 3, "y": 1, "color": self.colors['active']},
            {"name": "Knockout", "x": 3, "y": -1, "color": self.colors['accent']},
            {"name": "Training\nRetry", "x": 5, "y": -2, "color": self.colors['pending']},
            {"name": "Tournament\nWinner", "x": 9, "y": 0, "color": self.colors['success']},
            {"name": "Eliminated", "x": 8, "y": -2, "color": self.colors['failed']}
        ]
        
        for state in terminal_states:
            fig.add_trace(
                go.Scatter(
                    x=[state["x"]],
                    y=[state["y"]],
                    mode='markers+text',
                    marker=dict(
                        size=40,
                        color=state["color"],
                        line=dict(width=2, color='white')
                    ),
                    text=state["name"],
                    textposition="middle center",
                    textfont=dict(size=9, color='white', weight='bold'),
                    showlegend=False
                )
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Tournament Participant Flow<br><sub>Complete journey from registration to completion</sub>",
                x=0.5,
                font=dict(size=20, color=self.colors['text'])
            ),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            width=1400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 9.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 1.5]),
            font=dict(color=self.colors['text'])
        )
        
        return fig
    
    def create_scoring_pipeline_flow(self):
        """Create detailed scoring pipeline visualization"""
        
        # Define scoring stages
        scoring_stages = [
            {"name": "Training\nComplete", "x": 0, "y": 0, "color": self.colors['training']},
            {"name": "Test Data\nEvaluation", "x": 2, "y": 0, "color": self.colors['evaluation']},
            {"name": "Top 4\nSelection", "x": 4, "y": 0, "color": self.colors['accent']},
            {"name": "Synthetic\nEvaluation", "x": 6, "y": 0, "color": self.colors['evaluation']},
            {"name": "Loss\nValidation", "x": 8, "y": 0, "color": self.colors['secondary']},
            {"name": "Quality Score\nCalculation", "x": 10, "y": 0, "color": self.colors['success']},
            {"name": "Final\nRanking", "x": 12, "y": 0, "color": self.colors['primary']}
        ]
        
        fig = go.Figure()
        
        # Add scoring stages
        for i, stage in enumerate(scoring_stages):
            fig.add_trace(
                go.Scatter(
                    x=[stage["x"]],
                    y=[stage["y"]],
                    mode='markers+text',
                    marker=dict(
                        size=60,
                        color=stage["color"],
                        line=dict(width=3, color='white')
                    ),
                    text=stage["name"],
                    textposition="middle center",
                    textfont=dict(size=9, color='white', weight='bold'),
                    showlegend=False
                )
            )
            
            # Add arrows
            if i < len(scoring_stages) - 1:
                fig.add_annotation(
                    x=stage["x"] + 0.5,
                    y=stage["y"],
                    ax=scoring_stages[i+1]["x"] - 0.5,
                    ay=scoring_stages[i+1]["y"],
                    arrowhead=2,
                    arrowwidth=2,
                    arrowcolor=self.colors['text']
                )
        
        # Add detailed sub-processes
        sub_processes = [
            {"text": "• Test Loss Calculation\n• Initial Ranking\n• Performance Metrics", "x": 2, "y": -1.5},
            {"text": "• Sort by Test Performance\n• Select Top 4 Performers\n• Prepare for Synthetic Eval", "x": 4, "y": -1.5},
            {"text": "• Synthetic Dataset Eval\n• Cross-validation\n• Loss Comparison", "x": 6, "y": -1.5},
            {"text": "• Ratio Check (≤1.5)\n• NaN Handling\n• Validity Assessment", "x": 8, "y": -1.5},
            {"text": "• Weighted Loss Calculation\n• Task Work Score\n• Adjusted Score", "x": 10, "y": -1.5},
            {"text": "• Sigmoid Normalization\n• Final Score Assignment\n• Winner Determination", "x": 12, "y": -1.5}
        ]
        
        for process in sub_processes:
            fig.add_annotation(
                x=process["x"],
                y=process["y"],
                text=process["text"],
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=self.colors['text'],
                borderwidth=1,
                font=dict(size=8, color=self.colors['text'])
            )
        
        # Add decision points
        fig.add_trace(
            go.Scatter(
                x=[8],
                y=[1],
                mode='markers+text',
                marker=dict(
                    size=40,
                    color=self.colors['failed'],
                    symbol='diamond'
                ),
                text="Valid\nLosses?",
                textposition="middle center",
                textfont=dict(size=8, color='white', weight='bold'),
                showlegend=False
            )
        )
        
        # Add validation paths
        fig.add_annotation(x=8.3, y=0.7, ax=10, ay=0, arrowhead=2, arrowcolor='green', text="✓ Valid")
        fig.add_annotation(x=8.3, y=1.3, ax=2, ay=0, arrowhead=2, arrowcolor='red', text="✗ Test Only")
        
        fig.update_layout(
            title=dict(
                text="Tournament Scoring Pipeline<br><sub>Detailed evaluation and ranking process</sub>",
                x=0.5,
                font=dict(size=18, color=self.colors['text'])
            ),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500,
            width=1400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 13]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, 2]),
            font=dict(color=self.colors['text'])
        )
        
        return fig
    
    def create_tournament_progression_flow(self):
        """Create tournament progression and round advancement visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Tournament Types", "Round Progression", "Boss Round Mechanics", "GPU Allocation Flow"),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Tournament Types (top-left)
        tournament_types = ["TEXT Tournament", "IMAGE Tournament"]
        text_tasks = ["INSTRUCT", "DPO", "GRPO"]
        image_tasks = ["DIFFUSION"]
        
        fig.add_trace(
            go.Scatter(
                x=[0], y=[1],
                mode='markers+text',
                marker=dict(size=60, color=self.colors['primary']),
                text="TEXT<br>Tournament",
                textfont=dict(size=10, color='white', weight='bold'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        for i, task in enumerate(text_tasks):
            fig.add_trace(
                go.Scatter(
                    x=[1], y=[1 + (i-1)*0.5],
                    mode='markers+text',
                    marker=dict(size=30, color=self.colors['accent']),
                    text=task,
                    textfont=dict(size=8, color='white', weight='bold'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0],
                mode='markers+text',
                marker=dict(size=60, color=self.colors['secondary']),
                text="IMAGE<br>Tournament",
                textfont=dict(size=10, color='white', weight='bold'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[1], y=[0],
                mode='markers+text',
                marker=dict(size=30, color=self.colors['accent']),
                text="DIFFUSION",
                textfont=dict(size=8, color='white', weight='bold'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Round Progression (top-right)
        rounds = ["Group Stage", "Round of 16", "Quarterfinals", "Semifinals", "Boss Round"]
        for i, round_name in enumerate(rounds):
            color = self.colors['success'] if i == len(rounds)-1 else self.colors['primary']
            fig.add_trace(
                go.Scatter(
                    x=[i], y=[0],
                    mode='markers+text',
                    marker=dict(size=50, color=color),
                    text=round_name,
                    textfont=dict(size=8, color='white', weight='bold'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Boss Round Mechanics (bottom-left)
        boss_elements = [
            {"name": "Previous\nWinner", "x": 0, "y": 0, "color": self.colors['success']},
            {"name": "Challenger", "x": 2, "y": 0, "color": self.colors['primary']},
            {"name": "5% Victory\nMargin", "x": 1, "y": 1, "color": self.colors['accent']},
            {"name": "Best of 3\nTasks", "x": 1, "y": -1, "color": self.colors['secondary']}
        ]
        
        for element in boss_elements:
            fig.add_trace(
                go.Scatter(
                    x=[element["x"]], y=[element["y"]],
                    mode='markers+text',
                    marker=dict(size=40, color=element["color"]),
                    text=element["name"],
                    textfont=dict(size=8, color='white', weight='bold'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # GPU Allocation Flow (bottom-right)
        gpu_flow = [
            {"name": "Model\nSize", "x": 0, "y": 0, "color": self.colors['text']},
            {"name": "GPU\nThreshold", "x": 1, "y": 0, "color": self.colors['accent']},
            {"name": "Resource\nAllocation", "x": 2, "y": 0, "color": self.colors['primary']}
        ]
        
        for element in gpu_flow:
            fig.add_trace(
                go.Scatter(
                    x=[element["x"]], y=[element["y"]],
                    mode='markers+text',
                    marker=dict(size=40, color=element["color"]),
                    text=element["name"],
                    textfont=dict(size=8, color='white', weight='bold'),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=dict(
                text="Tournament Progression Overview<br><sub>Types, rounds, mechanics, and resource allocation</sub>",
                x=0.5,
                font=dict(size=18, color=self.colors['text'])
            ),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=800,
            width=1200,
            font=dict(color=self.colors['text'])
        )
        
        # Update all subplot axes
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
        
        return fig


def generate_participant_flow_charts(output_dir="output"):
    """Generate all participant flow charts"""
    
    visualizer = ParticipantFlowVisualizer()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate lifecycle flow
    lifecycle_fig = visualizer.create_participant_lifecycle_flow()
    lifecycle_fig.write_html(f"{output_dir}/participant_lifecycle_flow.html")
    lifecycle_fig.write_image(f"{output_dir}/participant_lifecycle_flow.png", width=1400, height=600)
    print("Generated participant lifecycle flow visualization")
    
    # Generate scoring pipeline
    scoring_fig = visualizer.create_scoring_pipeline_flow()
    scoring_fig.write_html(f"{output_dir}/scoring_pipeline_flow.html")
    scoring_fig.write_image(f"{output_dir}/scoring_pipeline_flow.png", width=1400, height=500)
    print("Generated scoring pipeline flow visualization")
    
    # Generate tournament progression
    progression_fig = visualizer.create_tournament_progression_flow()
    progression_fig.write_html(f"{output_dir}/tournament_progression_flow.html")
    progression_fig.write_image(f"{output_dir}/tournament_progression_flow.png", width=1200, height=800)
    print("Generated tournament progression flow visualization")


if __name__ == "__main__":
    generate_participant_flow_charts()