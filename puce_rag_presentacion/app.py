"""
Created by Analitika at 25/08/2024
contact@analitika.fr
"""
import reveal_slides as rs
from puce_rag_presentacion.slides import sample_markdown

theme = "white-contrast"
height = 500
content_height = 900
content_width = 1300
scale_range = [0.1, 3.0]
margin = 0
transition = "slide"
plugins = ["katex", "highlight", "zoom"]
hslidePos = 0
vslidePos = 0
fragPos = -1
overview = False
paused = False

# Add the streamlit-reveal-slide component to the Streamlit app.
currState = rs.slides(sample_markdown,
                    height=height,
                    theme=theme,
                    config={
                            "transition": transition,
                            "width": content_width,
                            "height": content_height,
                            "minScale": scale_range[0],
                            "center": True,
                            "maxScale": scale_range[1],
                            "margin": margin,
                            "plugins": plugins
                            },
                    initial_state={
                                    "indexh": hslidePos,
                                    "indexv": vslidePos,
                                    "indexf": fragPos,
                                    "paused": paused,
                                    "overview": overview
                                    },
                    markdown_props={"data-separator-vertical":"^--$"},
                    key="foo")
