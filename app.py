import json
import os
import time

import cv2
import numpy as np
import potrace
import streamlit as st
from tempfile import NamedTemporaryFile

# Page configuration
st.set_page_config(
    layout="wide",  # Use wide screen mode
    initial_sidebar_state="expanded"
)

# Initialize session_state to store processed data
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'video_frames' not in st.session_state:
    st.session_state.video_frames = None
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False

# Configuration parameters
st.sidebar.title("DesmosArt")
MEDIA_TYPE = st.sidebar.radio("Select media type:", ["Image", "Video"])
COLOUR = st.sidebar.color_picker("Line color", "#2464b4")
BILATERAL_FILTER = st.sidebar.checkbox("Bilateral filter", False)
USE_L2_GRADIENT = st.sidebar.checkbox("Use L2 gradient", False)
SHOW_GRID = st.sidebar.checkbox("Show grid", True)
SHOW_EXPRESSIONS = st.sidebar.checkbox("Show expression content", True)

if MEDIA_TYPE == "Video":
    st.sidebar.subheader("Video Settings")
    FPS = st.sidebar.slider("Frame rate", 1, 60, 15)  # Default reduced to 15fps
    FRAME_INTERVAL = 1000 // FPS  # Milliseconds
    
    # Add performance optimization options
    st.sidebar.subheader("Performance Optimization")
    VIDEO_SCALE = st.sidebar.slider("Video scale factor", 0.1, 1.0, 0.5, 0.1)  # Default scale to 50%
    MAX_EXPRESSIONS_PER_FRAME = st.sidebar.slider("Max expressions per frame", 100, 5000, 1000)  # Limit expressions per frame
    FRAME_SKIP = st.sidebar.slider("Frame interval", 1, 10, 2)  # Process every Nth frame

# Set maximum number of expressions
MAX_EXPRESSIONS = 50000
MAX_SIZE = 600

def resize_image(image, max_size=MAX_SIZE):
    if MEDIA_TYPE == "Video":
        # Use scale factor for videos
        height, width = image.shape[:2]
        new_width = int(width * VIDEO_SCALE)
        new_height = int(height * VIDEO_SCALE)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    else:
        # Image processing remains unchanged
        height, width = image.shape[:2]
        if max(height, width) <= max_size:
            return image
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def get_contours(image, nudge=.33):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if BILATERAL_FILTER:
        median = max(10, min(245, np.median(gray)))
        lower = int(max(0, (1 - nudge) * median))
        upper = int(min(255, (1 + nudge) * median))
        filtered = cv2.bilateralFilter(gray, 5, 50, 50)
        edged = cv2.Canny(filtered, lower, upper, L2gradient=USE_L2_GRADIENT)
    else:
        edged = cv2.Canny(gray, 30, 200)

    return edged[::-1]

def get_trace(data):
    for i in range(len(data)):
        data[i][data[i] > 1] = 1
    bmp = potrace.Bitmap(data)
    path = bmp.trace(2, potrace.TURNPOLICY_MINORITY, 1.0, 1, .5)
    return path

def get_latex(image):
    latex = []
    path = get_trace(get_contours(image))

    for curve in path.curves:
        segments = curve.segments
        start = curve.start_point
        for segment in segments:
            x0, y0 = start
            if segment.is_corner:
                x1, y1 = segment.c
                x2, y2 = segment.end_point
                latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x0, x1, y0, y1))
                latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x1, x2, y1, y2))
            else:
                x1, y1 = segment.c1
                x2, y2 = segment.c2
                x3, y3 = segment.end_point
                bezier_x = f"(1-t)^3*{x0}+3*(1-t)^2*t*{x1}+3*(1-t)*t^2*{x2}+t^3*{x3}"
                bezier_y = f"(1-t)^3*{y0}+3*(1-t)^2*t*{y1}+3*(1-t)*t^2*{y2}+t^3*{y3}"
                latex.append(f"({bezier_x},{bezier_y})")
            start = segment.end_point
    return latex

def get_expressions(image):
    exprs = []
    latex_expressions = get_latex(image)
    print(f"[INFO] Retrieved {len(latex_expressions)} expressions")
    
    # Choose maximum expressions based on media type
    max_expr = MAX_EXPRESSIONS_PER_FRAME if MEDIA_TYPE == "Video" else MAX_EXPRESSIONS
    
    for i, expr in enumerate(latex_expressions[:max_expr]):
        exprs.append({
            'id': f'expr{i}',
            'latex': expr,
            'color': COLOUR,
            'secret': not SHOW_EXPRESSIONS,
            'hidden': False
        })
    
    if len(latex_expressions) > max_expr:
        exprs.append({
            'id': 'truncated',
            'latex': f'\\text{{Showing {max_expr}/{len(latex_expressions)} expressions}}',
            'color': '#FF0000',
            'secret': False
        })
    
    if len(exprs) == 0:
        exprs.append({
            'id': 'empty',
            'latex': '\\text{No expressions}',
            'color': COLOUR,
            'secret': False
        })
    
    return exprs

@st.cache_data
def process_image(image):
    try:
        print("[INFO] Starting image processing...")
        expressions = get_expressions(image)
        print(f"[INFO] Processing complete, generated {len(expressions)} expressions")
        return expressions
    except Exception as e:
        import traceback
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        return []

def process_video(video_path):
    """Process video file, extract frames and generate expressions"""
    frames_data = []
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    frame_count = 0
    processed_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip frames according to frame interval
        if frame_count % FRAME_SKIP == 0:
            # Resize the frame
            frame = resize_image(frame)
            
            # Process frame and get expressions
            expressions = process_image(frame)
            frames_data.append(expressions)
            processed_count += 1
            
        # Update progress bar
        progress = int((frame_count + 1) / total_frames * 100)
        progress_bar.progress(progress)
        frame_count += 1
        
    cap.release()
    
    # Display processing information
    st.sidebar.info(f"Total frames: {total_frames}, Processed frames: {processed_count}")
    return frames_data

# Main program
if MEDIA_TYPE == "Image":
    uploaded_file = st.sidebar.file_uploader("Select image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image size
        height, width = image.shape[:2]
        st.sidebar.text(f"Original image size: {width}×{height}")
        
        # Compress image
        image = resize_image(image)
        
        # Display compressed image size
        height, width = image.shape[:2]
        st.sidebar.text(f"Processing size: {width}×{height}")
        
        # Display uploaded image (using compressed version)
        st.sidebar.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Uploaded image', use_container_width=True)
        
        if st.sidebar.button("Start Processing"):
            with st.spinner("Processing..."):
                start_time = time.time()
                st.session_state.processed_data = process_image(image)
                st.session_state.video_frames = None  # Clear video data
                process_time = time.time() - start_time
                st.sidebar.success(f"Processing complete! Took {process_time:.1f} seconds")
                
                # Display total curves
                if st.session_state.processed_data:
                    total_curves = len(st.session_state.processed_data)
                    st.sidebar.info(f"Generated {total_curves} Bezier curves")

else:  # Video processing
    uploaded_file = st.sidebar.file_uploader(
        "Select video",
        type=["mov", "mp4", "avi", "mkv", "MOV", "MP4", "AVI", "MKV"],
        help="Supports MOV, MP4, AVI, MKV and other video formats"
    )
    
    if uploaded_file is not None:
        # Get file extension and convert to lowercase
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Verify file extension
        allowed_extensions = ['.mov', '.mp4', '.avi', '.mkv']
        if file_extension not in allowed_extensions:
            st.error(f"Unsupported file format: {file_extension}. Please upload a MOV, MP4, AVI or MKV format video.")
            st.stop()
        
        # Save uploaded video to temporary file
        with NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        try:
            # Try to preview video
            video_bytes = uploaded_file.getvalue()
            st.sidebar.video(video_bytes)
            
            # Get video information
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Cannot open video file, please ensure the video format is correct")
                os.unlink(video_path)
                st.stop()
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps
            cap.release()
            
            # Display video information
            st.sidebar.text(f"Video information:")
            st.sidebar.text(f"Total frames: {total_frames}")
            st.sidebar.text(f"Original frame rate: {fps} FPS")
            st.sidebar.text(f"Duration: {duration:.1f} seconds")
            
            if st.sidebar.button("Start Processing"):
                with st.spinner("Processing video..."):
                    start_time = time.time()
                    st.session_state.video_frames = process_video(video_path)
                    st.session_state.processed_data = st.session_state.video_frames[0]  # Display first frame
                    st.session_state.current_frame = 0
                    process_time = time.time() - start_time
                    st.sidebar.success(f"Processing complete! Took {process_time:.1f} seconds")
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
        finally:
            # Delete temporary file
            if os.path.exists(video_path):
                os.unlink(video_path)

# Custom CSS
st.markdown("""
<style>
    /* Reset base styles */
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }

    /* Hide header and footer */
    header[data-testid="stHeader"],
    footer {
        display: none !important;
    }

    /* Sidebar styles */
    [data-testid="stSidebar"] {
        position: fixed !important;
        left: 0 !important;
        top: 0 !important;
        bottom: 0 !important;
        width: 350px !important;
        background: white !important;
    }

    [data-testid="stSidebar"] > div {
        padding: 1rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
        padding-bottom: 1rem !important;
    }

    /* Restore sidebar component styles */
    [data-testid="stSidebar"] .element-container {
        margin: 0.5rem 0 !important;
    }

    [data-testid="stSidebar"] .stRadio > div {
        margin: 0.5rem 0 !important;
    }

    [data-testid="stSidebar"] .stCheckbox > label {
        padding: 0.5rem 0 !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        margin: 0.5rem 0 !important;
    }

    /* Main content area styles */
    .main {
        position: absolute !important;
        left: 350px !important;
        right: 0 !important;
        top: 0 !important;
        bottom: 0 !important;
        background: transparent !important;
    }

    /* Desmos container styles */
    #calculator {
        position: absolute !important;
        left: 0 !important;
        top: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100% !important;
        height: 100% !important;
        border: none !important;
    }

    /* Remove scrollbars */
    ::-webkit-scrollbar {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# If an image or video has been processed, display the result
if st.session_state.processed_data is not None:
    # Generate Desmos embed code
    desmos_html = f"""
    <div id="calculator" style="position: absolute !important; left: 350px !important; top: 0 !important; right: 0 !important; bottom: 0 !important; width: calc(100% - 350px) !important; height: 100% !important; border: none !important;"></div>
    <script src="https://www.desmos.com/api/v1.8/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>
    <script>
        var elt = document.getElementById('calculator');
        var calculator = Desmos.GraphingCalculator(elt, {{
            expressions: true,
            settingsMenu: true,
            zoomButtons: true,
            expressionsTopbar: true,
            border: false,
            lockViewport: false,
            showGrid: {str(SHOW_GRID).lower()},
            expressionsCollapsed: false,
            squareAxes: true,
            autosize: true,
            projectorMode: true,
            aspectRatio: 1
        }});

        // Set expressions panel position
        calculator.updateSettings({{
            expressionsOrigin: {{ x: "left" }}
        }});
        
        // Set view bounds
        calculator.setMathBounds({{
            left: -10,
            right: 10,
            bottom: -10,
            top: 10
        }});
        
        // Listen for window resize
        window.addEventListener('resize', function() {{
            // Keep the view bounds unchanged, let Desmos handle scaling
            calculator.setMathBounds({{
                left: -10,
                right: 10,
                bottom: -10,
                top: 10
            }});
        }});
        
        var currentFrame = 0;
        var isPlaying = false;
        var frameInterval = {FRAME_INTERVAL if MEDIA_TYPE == "Video" else 0};
        var videoFrames = {json.dumps(st.session_state.video_frames) if st.session_state.video_frames else 'null'};
        
        function loadFrame(frameData) {{
            calculator.removeExpressions(calculator.getExpressions());
            frameData.forEach(function(expr) {{
                calculator.setExpression(expr);
            }});
        }}
        
        function playAnimation() {{
            if (!videoFrames || !isPlaying) return;
            
            loadFrame(videoFrames[currentFrame]);
            currentFrame = (currentFrame + 1) % videoFrames.length;
            
            setTimeout(playAnimation, frameInterval);
        }}
        
        try {{
            var expressions = {json.dumps(st.session_state.processed_data)};
            
            if (videoFrames) {{
                // Video mode
                isPlaying = true;
                playAnimation();
            }} else {{
                // Image mode
                function loadExpressionsInBatches(exprs, batchSize=500, delay=100) {{
                    var i = 0;
                    function loadBatch() {{
                        var batch = exprs.slice(i, i + batchSize);
                        if (batch.length > 0) {{
                            batch.forEach(function(expr) {{
                                calculator.setExpression(expr);
                            }});
                            i += batchSize;
                            if (i < exprs.length) {{
                                setTimeout(loadBatch, delay);
                            }} else {{
                                console.log("All loaded: " + exprs.length + " expressions");
                            }}
                        }}
                    }}
                    loadBatch();
                }}
                
                loadExpressionsInBatches(expressions, 100, 100);
            }}
            
        }} catch(e) {{
            console.error("Error loading expressions:", e);
        }}
    </script>
    """
    st.components.v1.html(desmos_html, height=1000, scrolling=False)
