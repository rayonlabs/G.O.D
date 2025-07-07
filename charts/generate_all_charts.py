#!/usr/bin/env python3
"""
G.O.D Tournament Visualization Generator
Generates all tournament structure and participant flow visualizations
"""

import os
import sys
import argparse
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tournament_structure import generate_tournament_structure_charts
from participant_flow import generate_participant_flow_charts


def create_index_html(output_dir):
    """Create an index.html file linking to all generated visualizations"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>G.O.D 5.0 Tournament Visualizations</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #2c3e50;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2E86AB;
                text-align: center;
                margin-bottom: 10px;
            }}
            .subtitle {{
                text-align: center;
                color: #7f8c8d;
                margin-bottom: 30px;
            }}
            .section {{
                margin-bottom: 40px;
            }}
            .section h2 {{
                color: #A23B72;
                border-bottom: 2px solid #A23B72;
                padding-bottom: 10px;
            }}
            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .chart-card {{
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                transition: transform 0.2s;
            }}
            .chart-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            .chart-card h3 {{
                color: #2E86AB;
                margin-bottom: 10px;
            }}
            .chart-card p {{
                color: #6c757d;
                margin-bottom: 15px;
            }}
            .btn {{
                display: inline-block;
                padding: 10px 20px;
                background: #2E86AB;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 5px;
                transition: background 0.2s;
            }}
            .btn:hover {{
                background: #1e5f7a;
            }}
            .btn-secondary {{
                background: #A23B72;
            }}
            .btn-secondary:hover {{
                background: #7a2d56;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #dee2e6;
                color: #6c757d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>G.O.D 5.0 Tournament Visualizations</h1>
            <p class="subtitle">The World's Greatest AutoML Script Competition</p>
            <p class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Tournament Structure Visualizations</h2>
                <p>These visualizations show the tournament bracket structures, group stages, and GPU allocation patterns.</p>
                <div class="chart-grid">
                    <div class="chart-card">
                        <h3>Small Text Tournament</h3>
                        <p>8 miners - Direct knockout format</p>
                        <a href="small_text_tournament.html" class="btn">View Interactive</a>
                        <a href="small_text_tournament.png" class="btn btn-secondary">Download PNG</a>
                    </div>
                    <div class="chart-card">
                        <h3>Large Text Tournament</h3>
                        <p>24 miners - Group stage format</p>
                        <a href="large_text_tournament.html" class="btn">View Interactive</a>
                        <a href="large_text_tournament.png" class="btn btn-secondary">Download PNG</a>
                    </div>
                    <div class="chart-card">
                        <h3>Image Tournament</h3>
                        <p>12 miners - Image generation tasks</p>
                        <a href="image_tournament.html" class="btn">View Interactive</a>
                        <a href="image_tournament.png" class="btn btn-secondary">Download PNG</a>
                    </div>
                    <div class="chart-card">
                        <h3>Mega Text Tournament</h3>
                        <p>32 miners - Large scale competition</p>
                        <a href="mega_text_tournament.html" class="btn">View Interactive</a>
                        <a href="mega_text_tournament.png" class="btn btn-secondary">Download PNG</a>
                    </div>
                    <div class="chart-card">
                        <h3>GPU Allocation</h3>
                        <p>Dynamic resource allocation by model size</p>
                        <a href="gpu_allocation.html" class="btn">View Interactive</a>
                        <a href="gpu_allocation.png" class="btn btn-secondary">Download PNG</a>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Participant Flow Visualizations</h2>
                <p>These visualizations show the complete participant journey from registration to tournament completion.</p>
                <div class="chart-grid">
                    <div class="chart-card">
                        <h3>Participant Lifecycle</h3>
                        <p>Complete journey from registration to winner determination</p>
                        <a href="participant_lifecycle_flow.html" class="btn">View Interactive</a>
                        <a href="participant_lifecycle_flow.png" class="btn btn-secondary">Download PNG</a>
                    </div>
                    <div class="chart-card">
                        <h3>Scoring Pipeline</h3>
                        <p>Detailed evaluation and ranking process</p>
                        <a href="scoring_pipeline_flow.html" class="btn">View Interactive</a>
                        <a href="scoring_pipeline_flow.png" class="btn btn-secondary">Download PNG</a>
                    </div>
                    <div class="chart-card">
                        <h3>Tournament Progression</h3>
                        <p>Overview of tournament types, rounds, and mechanics</p>
                        <a href="tournament_progression_flow.html" class="btn">View Interactive</a>
                        <a href="tournament_progression_flow.png" class="btn btn-secondary">Download PNG</a>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Usage Instructions</h2>
                <ul>
                    <li><strong>Interactive HTML:</strong> Click "View Interactive" for fully interactive charts with zoom, pan, and hover details</li>
                    <li><strong>PNG Downloads:</strong> Click "Download PNG" for high-resolution static images perfect for presentations</li>
                    <li><strong>Blog Integration:</strong> Use the PNG files for blog posts, or embed the HTML files for interactive content</li>
                    <li><strong>Customization:</strong> Modify the Python scripts to generate different tournament scenarios</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>Generated by G.O.D 5.0 Tournament Visualization System</p>
                <p>Powered by Plotly and Python</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html_content)


def main():
    """Main function to generate all visualizations"""
    
    parser = argparse.ArgumentParser(description="Generate G.O.D tournament visualizations")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--structure-only", action="store_true", help="Generate only tournament structure charts")
    parser.add_argument("--flow-only", action="store_true", help="Generate only participant flow charts")
    parser.add_argument("--no-index", action="store_true", help="Don't generate index.html")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"🏆 G.O.D 5.0 Tournament Visualization Generator")
    print(f"📁 Output directory: {args.output}")
    print(f"⚡ Starting visualization generation...")
    print()
    
    # Generate visualizations
    if not args.flow_only:
        print("🔄 Generating tournament structure visualizations...")
        generate_tournament_structure_charts(args.output)
        print()
    
    if not args.structure_only:
        print("🔄 Generating participant flow visualizations...")
        generate_participant_flow_charts(args.output)
        print()
    
    # Create index page
    if not args.no_index:
        print("📄 Creating index.html...")
        create_index_html(args.output)
        print()
    
    print("✅ All visualizations generated successfully!")
    print(f"🌐 Open {args.output}/index.html to view all charts")
    print()
    print("💡 Tips:")
    print("  - Use HTML files for interactive viewing")
    print("  - Use PNG files for blog posts and presentations")
    print("  - Customize the Python scripts for different scenarios")


if __name__ == "__main__":
    main()