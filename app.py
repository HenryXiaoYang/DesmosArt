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
if 'current_color' not in st.session_state:
    st.session_state.current_color = "#2464b4"  # Default color
if 'active_color' not in st.session_state:
    st.session_state.active_color = "#2464b4"  # Default color

# Configuration parameters
st.sidebar.title("DesmosArt")
MEDIA_TYPE = st.sidebar.radio("Select media type:", ["Image", "Video"])

# Initialize video variables
if MEDIA_TYPE == "Video":
    st.sidebar.subheader("Video Settings")
    if 'video_fps' not in st.session_state:
        st.session_state.video_fps = 30  # Default FPS if not set
    PLAYBACK_SPEED = st.sidebar.slider("Playback speed", 0.25, 2.0, 1.0, 0.25)
    FRAME_INTERVAL = int(1000 / (st.session_state.video_fps * PLAYBACK_SPEED))  # Convert FPS to milliseconds
    FRAME_SKIP = st.sidebar.slider("Process every Nth frame", 1, 10, 2)  # Process every Nth frame
    
    # Add performance optimization options
    st.sidebar.subheader("Performance Optimization")
    VIDEO_SCALE = st.sidebar.slider("Video scale factor", 0.1, 1.0, 0.5, 0.1)  # Default scale to 50%
    MAX_EXPRESSIONS_PER_FRAME = st.sidebar.slider("Max expressions per frame", 100, 5000, 1000)  # Limit expressions per frame
else:
    FRAME_INTERVAL = 0  # Not used for images
    FRAME_SKIP = 1  # Default
    VIDEO_SCALE = 1.0  # Default
    MAX_EXPRESSIONS_PER_FRAME = 5000  # Default

# Set up a dedicated color management system
COLOUR = st.sidebar.color_picker("Line color", st.session_state.active_color, key="color_picker")

# Add a button to update color only when processed data exists
if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
    if st.sidebar.button("Update Color"):
        # Store the new color in session state - using upper for consistency
        st.session_state.active_color = COLOUR.upper()
        st.session_state.current_color = COLOUR.upper()
        st.sidebar.success(f"Color updated to {COLOUR.upper()}!")
        # Force a rerun to ensure the UI updates
        st.experimental_rerun()
else:
    # Initialize current_color if needed
    st.session_state.active_color = COLOUR.upper()
    st.session_state.current_color = COLOUR.upper()

BILATERAL_FILTER = st.sidebar.checkbox("Bilateral filter", False)
USE_L2_GRADIENT = st.sidebar.checkbox("Use L2 gradient", False)
SHOW_GRID = st.sidebar.checkbox("Show grid", True)
SHOW_EXPRESSIONS = st.sidebar.checkbox("Show expression content", True)

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

def get_expressions(image, color):
    exprs = []
    latex_expressions = get_latex(image)
    print(f"[INFO] Retrieved {len(latex_expressions)} expressions")
    
    # Choose maximum expressions based on media type
    max_expr = MAX_EXPRESSIONS_PER_FRAME if MEDIA_TYPE == "Video" else MAX_EXPRESSIONS
    
    for i, expr in enumerate(latex_expressions[:max_expr]):
        exprs.append({
            'id': f'expr{i}',
            'latex': expr,
            'color': color.upper(),  # Convert to uppercase to ensure proper format
            'lineStyle': 'SOLID',  # Use string instead of undefined reference
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
            'color': color.upper(),
            'secret': False
        })
    
    return exprs

@st.cache_data
def process_image(image, color):
    try:
        print("[INFO] Starting image processing...")
        expressions = get_expressions(image, color)
        print(f"[INFO] Processing complete, generated {len(expressions)} expressions")
        return expressions
    except Exception as e:
        import traceback
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        return []

def process_video(video_path, color):
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
            expressions = process_image(frame, color)
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
                st.session_state.processed_data = process_image(image, COLOUR)
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
            st.session_state.video_fps = fps  # Store the FPS in session state
            duration = total_frames / fps
            cap.release()
            
            # Calculate the correct frame interval based on video FPS and playback speed
            FRAME_INTERVAL = int(1000 / (fps * PLAYBACK_SPEED))
            
            # Display video information
            st.sidebar.text(f"Video information:")
            st.sidebar.text(f"Total frames: {total_frames}")
            st.sidebar.text(f"Original frame rate: {fps} FPS")
            st.sidebar.text(f"Playback interval: {FRAME_INTERVAL}ms")
            st.sidebar.text(f"Duration: {duration:.1f} seconds")
            
            if st.sidebar.button("Start Processing"):
                with st.spinner("Processing video..."):
                    start_time = time.time()
                    st.session_state.video_frames = process_video(video_path, COLOUR)
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
        height: 100vh !important; /* Use viewport height */
        padding: 0 !important;
        margin: 0 !important;
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

    /* Make the iframe take full height */
    iframe {
        height: 100vh !important;
        width: 100% !important;
        position: absolute !important;
        left: 0 !important;
        top: 0 !important;
    }

    /* Remove scrollbars */
    ::-webkit-scrollbar {
        display: none !important;
    }

    /* Hide overflow */
    html, body {
        overflow: hidden !important;
        height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# If an image or video has been processed, display the result
if st.session_state.processed_data is not None:
    # Generate Desmos embed code
    desmos_html = f"""
    <div id="calculator" style="position: absolute !important; left: 350px !important; top: 0 !important; right: 0 !important; bottom: 0 !important; width: calc(100% - 350px) !important; height: 100vh !important; border: none !important;"></div>
    <script src="https://www.desmos.com/api/v1.8/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>
    <script>
        // Define custom colors palette
        var customColors = {{
            "CURRENT": "{st.session_state.current_color}",
            "RED": "#c74440",
            "BLUE": "#2d70b3",
            "GREEN": "#388c46",
            "PURPLE": "#6042a6",
            "ORANGE": "#fa7e19",
            "BLACK": "#000000"
        }};
        
        // Wait for document to be fully loaded
        window.onload = function() {{
            initializeCalculator();
        }};
        
        // Initialize the calculator immediately as well, in case the onload event doesn't fire
        initializeCalculator();
        
        // Initialize the calculator
        function initializeCalculator() {{
            // Check if calculator has already been initialized
            if (window.calculatorInitialized) return;
            window.calculatorInitialized = true;
        
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
                aspectRatio: 1,
                colors: customColors  // Apply custom colors palette
            }});
    
            // Set expressions panel position
            calculator.updateSettings({{
                expressionsOrigin: {{ x: "left" }},
                defaultColors: false  // Disable Desmos default colors
            }});
            
            // Set view bounds - ensuring a perfect square viewing area
            var size = 10;
            calculator.setMathBounds({{
                left: -size,
                right: size,
                bottom: -size,
                top: size
            }});
            
            // Ensure square axes are enforced
            calculator.updateSettings({{
                squareAxes: true
            }});
            
            // Force resize to ensure the aspect ratio is applied
            setTimeout(function() {{
                calculator.resize();
                // Double-check that square axes are enforced
                calculator.updateSettings({{
                    squareAxes: true
                }});
            }}, 100);
            
            // Listen for window resize
            window.addEventListener('resize', function() {{
                // Keep the view bounds unchanged, let Desmos handle scaling
                calculator.setMathBounds({{
                    left: -size,
                    right: size,
                    bottom: -size,
                    top: size
                }});
                // Ensure square axes are maintained on resize
                calculator.updateSettings({{
                    squareAxes: true
                }});
                // Force resize to apply the settings
                calculator.resize();
            }});
            
            var currentFrame = 0;
            var isPlaying = false;
            var frameInterval = {FRAME_INTERVAL};  // This is now properly calculated based on video FPS
            var videoFrames = {json.dumps(st.session_state.video_frames) if st.session_state.video_frames else 'null'};
            var currentColor = "{st.session_state.current_color}";
            
            function loadFrame(frameData) {{
                calculator.removeExpressions(calculator.getExpressions());
                frameData.forEach(function(expr) {{
                    // Update the color to the current color for all expressions except special ones
                    if (expr.id !== 'truncated' && !expr.color.startsWith('#FF')) {{
                        expr.color = currentColor;
                    }}
                    calculator.setExpression(expr);
                }});
            }}
            
            function playAnimation() {{
                if (!videoFrames || !isPlaying) return;
                
                loadFrame(videoFrames[currentFrame]);
                currentFrame = (currentFrame + 1) % videoFrames.length;
                
                // Use frameInterval for timing between frames
                setTimeout(playAnimation, frameInterval);
            }}
            
            // Add play/pause controls for video
            if (videoFrames) {{
                // Create playback controls
                var controlsDiv = document.createElement('div');
                controlsDiv.style.position = 'absolute';
                controlsDiv.style.bottom = '20px';
                controlsDiv.style.right = '20px';
                controlsDiv.style.zIndex = '1000';
                controlsDiv.style.background = 'rgba(255, 255, 255, 0.7)';
                controlsDiv.style.padding = '10px';
                controlsDiv.style.borderRadius = '5px';
                
                var playPauseBtn = document.createElement('button');
                playPauseBtn.innerHTML = '⏸️ Pause';
                playPauseBtn.style.marginRight = '10px';
                playPauseBtn.onclick = function() {{
                    isPlaying = !isPlaying;
                    if (isPlaying) {{
                        playPauseBtn.innerHTML = '⏸️ Pause';
                        playAnimation();
                    }} else {{
                        playPauseBtn.innerHTML = '▶️ Play';
                    }}
                }};
                
                // Add playback rate control info
                var infoSpan = document.createElement('span');
                infoSpan.innerHTML = 'Speed: {PLAYBACK_SPEED}x | ' + videoFrames.length + ' frames';
                
                controlsDiv.appendChild(playPauseBtn);
                controlsDiv.appendChild(infoSpan);
                document.body.appendChild(controlsDiv);
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
                                    // Update the color to the current color for all expressions except special ones
                                    if (expr.id !== 'truncated' && !expr.color.startsWith('#FF')) {{
                                        expr.color = currentColor;
                                    }}
                                    calculator.setExpression(expr);
                                }});
                                i += batchSize;
                                if (i < exprs.length) {{
                                    setTimeout(loadBatch, delay);
                                }} else {{
                                    console.log("All loaded: " + exprs.length + " expressions with color " + currentColor);
                                    // After loading all expressions, ensure the bounds are properly set
                                    calculator.setMathBounds({{
                                        left: -size,
                                        right: size,
                                        bottom: -size,
                                        top: size
                                    }});
                                    calculator.updateSettings({{
                                        squareAxes: true
                                    }});
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
        }}
    </script>
    """
    
    # Use height="100%" instead of a fixed value to allow full height rendering
    st.components.v1.html(desmos_html, height=800, scrolling=False)
    
    # Add JavaScript to dynamically adjust the component height
    st.markdown("""
    <script>
        // Find iframe and set its height to full viewport height
        window.addEventListener('load', function() {
            setTimeout(function() {
                var iframes = document.getElementsByTagName('iframe');
                for (var i = 0; i < iframes.length; i++) {
                    iframes[i].style.height = '100vh';
                    iframes[i].style.minHeight = '100vh';
                    iframes[i].parentNode.style.height = '100vh';
                    iframes[i].parentNode.style.minHeight = '100vh';
                }
            }, 100);
        });
        
        // Adjust on resize
        window.addEventListener('resize', function() {
            var iframes = document.getElementsByTagName('iframe');
            for (var i = 0; i < iframes.length; i++) {
                iframes[i].style.height = '100vh';
                iframes[i].style.minHeight = '100vh';
            }
        });
    </script>
    """, unsafe_allow_html=True)
