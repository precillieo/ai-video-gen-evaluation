import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

class Visualizer:
    def plot_comparison_chart(self, data, title, ylabel):
        """Plot comparison chart for metrics.
        
        Args:
            data: Dictionary of metrics where each value can be either:
                 - A single value (float)
                 - A dictionary of values for multiple metrics
            title: Chart title
            ylabel: Y-axis label
        """
        plt.figure(figsize=(10, 6))
        
        # If data is a dictionary of dictionaries (multiple metrics)
        if isinstance(next(iter(data.values())), dict):
            # Get all unique metric names
            metric_names = set()
            for metrics in data.values():
                metric_names.update(metrics.keys())
            metric_names = sorted(list(metric_names))
            
            # Prepare data for grouped bar chart
            x = np.arange(len(data))
            width = 0.8 / len(metric_names)
            
            # Plot each metric
            for i, metric in enumerate(metric_names):
                values = [metrics.get(metric, 0) for metrics in data.values()]
                plt.bar(x + i * width, values, width, label=metric)
            
            plt.xticks(x + width * (len(metric_names) - 1) / 2, data.keys(), rotation=45)
            plt.legend()
        else:
            # Single metric case
            plt.bar(data.keys(), data.values())
            plt.xticks(rotation=45)
        
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        
        return plt.gcf()

    @staticmethod
    def display_results(results, model_name, evaluator):
        st.subheader(model_name)
        st.image(results["image"], use_container_width=True)
        
        st.metric("CLIP Score", f"{results['clip_score']:.2f}%")
        st.text_area("BLIP Caption", results["blip_caption"], height=100)

    @staticmethod
    def display_video_results(video_name, frames, clip_score, blip_caption):
        st.subheader(f"Video: {video_name}")
        st.image(frames[0], caption="First frame", use_container_width=True)
        st.metric("CLIP Score", f"{clip_score:.2f}%")
        st.text_area("BLIP Caption (first frame)", blip_caption, height=100) 