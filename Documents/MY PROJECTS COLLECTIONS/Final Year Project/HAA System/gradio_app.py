"""
Gradio interface for HAA
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from config import *
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from visualization import Visualizer
from prediction import PredictionEngine

class HumanitarianAidApp:
    def __init__(self):
        self.model_trainer = ModelTrainer()
        self.visualizer = Visualizer()
        self.prediction_engine = PredictionEngine()
        self.preprocessor = DataPreprocessor()
        self.df = None
        
        # Load or train model
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the system with data and model"""
        try:
            # Load processed data
            if PROCESSED_DATA_FILE.exists():
                self.df = pd.read_csv(PROCESSED_DATA_FILE)
            else:
                # Process data if not available
                _, self.df = self.preprocessor.preprocess_pipeline()
            
            # Load or train model
            if not self.model_trainer.load_model():
                print("Training new model...")
                from model_training import train_and_save_model
                self.model_trainer, _, _ = train_and_save_model()
            
            self.prediction_engine.load_models()
            print("System initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing system: {e}")
            # Create sample data as fallback
            self.df = self.preprocessor.create_sample_data()
    
    def predict_funding_needs(self, sector_name, year, quarter, population_in_need, 
                            people_reached, country="South Sudan"):
        """Predict funding needs based on input parameters"""
        try:
            # Prepare input data
            input_data = {
                'Country': country,
                'Sector_Name': sector_name,
                'Year': int(year),
                'Quarter': int(quarter),
                'Population_In_Need': float(population_in_need),
                'People_Reached': float(people_reached)
            }
            
            # Get prediction
            #result = self.prediction_engine.predict_funding_needs(input_data)
            result = self.prediction_engine.predict_funding_requirement(
                country=input_data['Country_encoded'],
                sector_name=input_data['Sector_Name_encoded'],
                Plan_name=input_data['Plan_Name_encoded'],
                year=input_data['Year'],
                quarter=input_data['Quarter'],
                population_in_need=input_data['Population_In_Need'],
                people_reached=input_data['People_Reached'])
            
            # Format output
            prediction_text = f"""
            # Funding Prediction Results
            
            Sector: {sector_name}
            Year: {year} - Q{quarter}
            Population in Need: {population_in_need:,}
            People to Reach: {people_reached:,}
            
            # Predicted Requirements:
            Amount: ${result['prediction']:,.2f}
            Confidence Interval:  ${result['confidence_interval'][0]:,.2f} - ${result['confidence_interval'][1]:,.2f}
            
            # Recommendations:
            {result['recommendations']}
            """
            
            return prediction_text
            
        except Exception as e:
            return f"Error making prediction: {str(e)}"
    
    def analyze_sector_performance(self, selected_sectors):
        """Analyze performance of selected sectors"""
        try:
            if not selected_sectors:
                return "Please select at least one sector for analysis."
            
            # Filter data for selected sectors
            filtered_df = self.df[self.df['Sector_Name'].isin(selected_sectors)]
            
            if filtered_df.empty:
                return "No data available for selected sectors."
            
            # Create performance visualization
            fig = self.visualizer.plot_sector_performance(filtered_df)
            
            # Calculate summary statistics
            summary_stats = filtered_df.groupby('Sector_Name').agg({
                'Total_Requirements': ['sum', 'mean'],
                'Total_Funding': ['sum', 'mean'],
                'Funding_Percentage': 'mean',
                'People_Reached': 'sum'
            }).round(2)
            
            return fig, summary_stats.to_string()
            
        except Exception as e:
            return f"Error analyzing sector performance: {str(e)}", ""
    
    def upload_and_analyze_data(self, file):
        """Upload and analyze custom data file"""
        try:
            if file is None:
                return "Please upload a CSV file.", None
            
            # Read uploaded file
            df_new = pd.read_csv(file.name)
            
            # Basic validation
            required_columns = ['Sector_Name', 'Total_Requirements', 'Total_Funding']
            missing_columns = [col for col in required_columns if col not in df_new.columns]
            
            if missing_columns:
                return f"Missing required columns: {missing_columns}", None
            
            # Create visualization for uploaded data
            fig = self.visualizer.plot_funding_by_sector(df_new)
            
            # Generate summary
            summary = f"""
            ## Data Upload Summary
            
            Total Records: {len(df_new)}
            Sectors: {df_new['Sector_Name'].nunique()}
            Total Requirements: ${df_new['Total_Requirements'].sum():,.2f}
            Total Funding: ${df_new['Total_Funding'].sum():,.2f}
            Average Funding Rate: {(df_new['Total_Funding'].sum() / df_new['Total_Requirements'].sum() * 100):.1f}%
            
            ### Sectors Included:
            {', '.join(df_new['Sector_Name'].unique())}
            """
            
            return summary, fig
            
        except Exception as e:
            return f"Error processing uploaded file: {str(e)}", None
    
    def generate_allocation_recommendations(self, total_budget, priority_sectors):
        """Generate resource allocation recommendations"""
        try:
            if not priority_sectors:
                return "Please select priority sectors for allocation."
            
            # Filter data for priority sectors
            priority_df = self.df[self.df['Sector_Name'].isin(priority_sectors)]
            
            if priority_df.empty:
                return "No data available for selected priority sectors."
            
            # Calculate allocation based on historical needs and performance
            sector_stats = priority_df.groupby('Sector_Name').agg({
                'Total_Requirements': 'mean',
                'Funding_Percentage': 'mean',
                'People_Reached': 'sum',
                'Efficiency_Ratio': 'mean'
            }).reset_index()
            
            # Calculate allocation weights
            total_requirements = sector_stats['Total_Requirements'].sum()
            sector_stats['Allocation_Weight'] = sector_stats['Total_Requirements'] / total_requirements
            sector_stats['Recommended_Allocation'] = sector_stats['Allocation_Weight'] * float(total_budget)
            
            # Adjust based on efficiency
            efficiency_factor = sector_stats['Efficiency_Ratio'] / sector_stats['Efficiency_Ratio'].max()
            sector_stats['Adjusted_Allocation'] = sector_stats['Recommended_Allocation'] * (1 + efficiency_factor * 0.2)
            
            # Normalize to total budget
            allocation_sum = sector_stats['Adjusted_Allocation'].sum()
            sector_stats['Final_Allocation'] = sector_stats['Adjusted_Allocation'] * (float(total_budget) / allocation_sum)
            
            # Create visualization
            fig = go.Figure(data=go.Bar(
                x=sector_stats['Sector_Name'],
                y=sector_stats['Final_Allocation'],
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title=f"Recommended Budget Allocation (Total: ${float(total_budget):,.2f})",
                xaxis_title="Sectors",
                yaxis_title="Allocated Amount ($)",
                xaxis_tickangle=45
            )
            
            # Generate recommendations text
            recommendations = f"""
            ## Resource Allocation Recommendations
            
            Total Budget: ${float(total_budget):,.2f}
            Priority Sectors: {len(priority_sectors)}
            
            ### Allocation Breakdown:
            """
            
            for _, row in sector_stats.iterrows():
                recommendations += f"""
                {row['Sector_Name']}:
                - Recommended Amount: ${row['Final_Allocation']:,.2f} ({row['Final_Allocation']/float(total_budget)*100:.1f}%)
                - Historical Average Need: ${row['Total_Requirements']:,.2f}
                - Average Funding Rate: {row['Funding_Percentage']:.1f}%
                - Efficiency Score: {row['Efficiency_Ratio']:.2f}
                """
            
            return recommendations, fig
            
        except Exception as e:
            return f"Error generating recommendations: {str(e)}", None
    
    def create_gradio_interface(self):
        """Create the main Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #3b5998 0%, #1e3c72 100%);
            color: #f9f9f9;
            padding: 40px;
            border-radius: 12px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            margin: 50px;
        }

        .main-header h1 {
            font-size: 34px;
            margin-bottom: 16px;
            font-weight: bold;
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }

        .main-header p {
            font-size: 18px;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            color: #e2e2e2;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.6);
        }

        .metric-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            margin: 10px 50px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        """
        
        with gr.Blocks(css=css, title=GRADIO_TITLE) as interface:
            
            # Header
            with gr.Row():
                gr.HTML(f"""
                <div class="main-header">
                    <h1>{GRADIO_TITLE}</h1>
                    <p>{GRADIO_DESCRIPTION}</p>
                </div>
                """)
            
            # Main tabs
            with gr.Tabs():
                
                # Dashboard Tab
                with gr.TabItem("📈 Dashboard"):
                    gr.Markdown("## Humanitarian Aid Dashboard")
                    
                    if self.df is not None:
                        # Summary metrics
                        total_req = self.df['Total_Requirements'].sum()
                        total_fund = self.df['Total_Funding'].sum()
                        funding_rate = (total_fund / total_req * 100) if total_req > 0 else 0
                        
                        gr.HTML(f"""
                        <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                            <div class="metric-box">
                                <h3>Total Requirements</h3>
                                <h2>${total_req:,.0f}</h2>
                            </div>
                            <div class="metric-box">
                                <h3>Total Funding</h3>
                                <h2>${total_fund:,.0f}</h2>
                            </div>
                            <div class="metric-box">
                                <h3>Funding Rate</h3>
                                <h2>{funding_rate:.1f}%</h2>
                            </div>
                        </div>
                        """)
                        
                        # Dashboard visualizations
                        with gr.Row():
                            funding_dist_plot = gr.Plot(
                                value=self.visualizer.plot_funding_by_sector(self.df)
                            )
                        
                        with gr.Row():
                            trends_plot = gr.Plot(
                                value=self.visualizer.plot_funding_trends(self.df)
                            )
                
                # Prediction Tab
                with gr.TabItem("💰 Funding Prediction"):
                    gr.Markdown("## Predict Humanitarian Funding Needs")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            sector_dropdown = gr.Dropdown(
                                choices=list(self.df['Sector_Name'].unique()) if self.df is not None else [],
                                label="Select Sector",
                                value="Education"
                            )
                            year_slider = gr.Slider(
                                minimum=2024,
                                maximum=2026,
                                value=2025,
                                step=1,
                                label="Year"
                            )
                            quarter_slider = gr.Slider(
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1,
                                label="Quarter"
                            )
                            population_input = gr.Number(
                                value=1000000,
                                label="Population in Need"
                            )
                            people_reached_input = gr.Number(
                                value=500000,
                                label="Target People to Reach"
                            )
                            predict_button = gr.Button("Predict Funding Needs", variant="primary")
                        
                        with gr.Column(scale=2):
                            prediction_output = gr.Markdown()
                    
                    predict_button.click(
                        fn=self.predict_funding_needs,
                        inputs=[sector_dropdown, year_slider, quarter_slider, 
                               population_input, people_reached_input],
                        outputs=prediction_output
                    )
                
                # Analysis Tab
                with gr.TabItem("📊 Sector Analysis"):
                    gr.Markdown("## Analyze Sector Performance")
                    
                    with gr.Row():
                        sector_checkboxes = gr.CheckboxGroup(
                            choices=list(self.df['Sector_Name'].unique()) if self.df is not None else [],
                            label="Select Sectors to Analyze",
                            value=["Education", "Health"]
                        )
                        analyze_button = gr.Button("Analyze Performance", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            analysis_plot = gr.Plot()
                        with gr.Column():
                            analysis_stats = gr.Textbox(
                                label="Performance Statistics",
                                lines=10
                            )
                    
                    analyze_button.click(
                        fn=self.analyze_sector_performance,
                        inputs=sector_checkboxes,
                        outputs=[analysis_plot, analysis_stats]
                    )
                
                # Data Upload Tab
                with gr.TabItem("📤 Data Upload"):
                    gr.Markdown("## Upload and Analyze Custom Data")
                    
                    with gr.Row():
                        file_upload = gr.File(
                            label="Upload CSV File",
                            file_types=[".csv"]
                        )
                        upload_button = gr.Button("Analyze Uploaded Data", variant="primary")
                    
                    with gr.Row():
                        with gr.Column():
                            upload_summary = gr.Markdown()
                        with gr.Column():
                            upload_plot = gr.Plot()
                    
                    upload_button.click(
                        fn=self.upload_and_analyze_data,
                        inputs=file_upload,
                        outputs=[upload_summary, upload_plot]
                    )
                
                # Resource Allocation Tab
                with gr.TabItem("🎯 Resource Allocation"):
                    gr.Markdown("## Optimize Resource Allocation")
                    
                    with gr.Row():
                        with gr.Column():
                            total_budget_input = gr.Number(
                                value=10000000,
                                label="Total Available Budget ($)"
                            )
                            priority_sectors = gr.CheckboxGroup(
                                choices=list(self.df['Sector_Name'].unique()) if self.df is not None else [],
                                label="Priority Sectors",
                                value=["Health", "Food Security and Livelihoods", "Education"]
                            )
                            allocate_button = gr.Button("Generate Allocation Plan", variant="primary")
                        
                        with gr.Column(scale=2):
                            allocation_recommendations = gr.Markdown()
                    
                    with gr.Row():
                        allocation_plot = gr.Plot()
                    
                    allocate_button.click(
                        fn=self.generate_allocation_recommendations,
                        inputs=[total_budget_input, priority_sectors],
                        outputs=[allocation_recommendations, allocation_plot]
                    )
                
                
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; padding: 20px; color: #666;">
                <p>South Sudan Humanitarian Aid Analytics and Prediction System | HAA System</p>
            </div>
            """)
        
        return interface

def create_app():
    """Create and return the Gradio app"""
    app = HumanitarianAidApp()
    return app.create_gradio_interface()

if __name__ == "__main__":
    app = create_app()
    app.launch(share=True, debug=True)