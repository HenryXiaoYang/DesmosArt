import json
import os
import time

import cv2
import numpy as np
import potrace
import streamlit as st
from tempfile import NamedTemporaryFile

# 设置页面配置
st.set_page_config(
    layout="wide",  # 使用宽屏模式
    initial_sidebar_state="expanded"
)

# 初始化session_state以存储处理后的数据
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'video_frames' not in st.session_state:
    st.session_state.video_frames = None
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False

# 配置参数
st.sidebar.title("贝塞尔曲线渲染器")
MEDIA_TYPE = st.sidebar.radio("选择媒体类型", ["图片", "视频"])
COLOUR = st.sidebar.color_picker("线条颜色", "#2464b4")
BILATERAL_FILTER = st.sidebar.checkbox("双边滤波", False)
USE_L2_GRADIENT = st.sidebar.checkbox("使用L2梯度", False)
SHOW_GRID = st.sidebar.checkbox("显示网格", True)
SHOW_EXPRESSIONS = st.sidebar.checkbox("显示表达式内容", True)

if MEDIA_TYPE == "视频":
    st.sidebar.subheader("视频设置")
    FPS = st.sidebar.slider("帧率", 1, 60, 15)  # 默认降低到15fps
    FRAME_INTERVAL = 1000 // FPS  # 毫秒
    
    # 添加性能优化选项
    st.sidebar.subheader("性能优化")
    VIDEO_SCALE = st.sidebar.slider("视频缩放比例", 0.1, 1.0, 0.5, 0.1)  # 默认缩放到50%
    MAX_EXPRESSIONS_PER_FRAME = st.sidebar.slider("每帧最大表达式数", 100, 5000, 1000)  # 限制每帧的表达式数量
    FRAME_SKIP = st.sidebar.slider("帧间隔", 1, 10, 2)  # 每隔几帧处理一帧

# 设置最大表达式数量
MAX_EXPRESSIONS = 50000
MAX_SIZE = 600

def resize_image(image, max_size=MAX_SIZE):
    if MEDIA_TYPE == "视频":
        # 对视频使用缩放比例
        height, width = image.shape[:2]
        new_width = int(width * VIDEO_SCALE)
        new_height = int(height * VIDEO_SCALE)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    else:
        # 图片处理保持不变
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
    print(f"[信息] 获取到 {len(latex_expressions)} 个表达式")
    
    # 根据媒体类型选择最大表达式数量
    max_expr = MAX_EXPRESSIONS_PER_FRAME if MEDIA_TYPE == "视频" else MAX_EXPRESSIONS
    
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
            'latex': f'\\text{{显示了 {max_expr}/{len(latex_expressions)} 个表达式}}',
            'color': '#FF0000',
            'secret': False
        })
    
    if len(exprs) == 0:
        exprs.append({
            'id': 'empty',
            'latex': '\\text{无表达式}',
            'color': COLOUR,
            'secret': False
        })
    
    return exprs

@st.cache_data
def process_image(image):
    try:
        print("[信息] 开始处理图片...")
        expressions = get_expressions(image)
        print(f"[信息] 处理完成，生成了 {len(expressions)} 个表达式")
        return expressions
    except Exception as e:
        import traceback
        print(f"处理图片时发生错误: {str(e)}")
        print(traceback.format_exc())
        return []

def process_video(video_path):
    """处理视频文件，提取帧并生成表达式"""
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
            
        # 根据帧间隔跳过一些帧
        if frame_count % FRAME_SKIP == 0:
            # 调整帧的大小
            frame = resize_image(frame)
            
            # 处理帧并获取表达式
            expressions = process_image(frame)
            frames_data.append(expressions)
            processed_count += 1
            
        # 更新进度条
        progress = int((frame_count + 1) / total_frames * 100)
        progress_bar.progress(progress)
        frame_count += 1
        
    cap.release()
    
    # 显示处理信息
    st.sidebar.info(f"总帧数: {total_frames}, 处理帧数: {processed_count}")
    return frames_data

# 主程序
if MEDIA_TYPE == "图片":
    uploaded_file = st.sidebar.file_uploader("选择图片", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # 读取图片
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 显示原始图片尺寸
        height, width = image.shape[:2]
        st.sidebar.text(f"原始图片尺寸: {width}×{height}")
        
        # 压缩图片
        image = resize_image(image)
        
        # 显示压缩后的图片尺寸
        height, width = image.shape[:2]
        st.sidebar.text(f"处理尺寸: {width}×{height}")
        
        # 显示上传的图片（使用压缩后的版本）
        st.sidebar.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='上传的图片', use_container_width=True)
        
        if st.sidebar.button("开始处理"):
            with st.spinner("处理中..."):
                start_time = time.time()
                st.session_state.processed_data = process_image(image)
                st.session_state.video_frames = None  # 清除视频数据
                process_time = time.time() - start_time
                st.sidebar.success(f"处理完成！耗时 {process_time:.1f} 秒")
                
                # 显示线条总数
                if st.session_state.processed_data:
                    total_curves = len(st.session_state.processed_data)
                    st.sidebar.info(f"共生成了 {total_curves} 条贝塞尔曲线")

else:  # 视频处理
    uploaded_file = st.sidebar.file_uploader(
        "选择视频",
        type=["mov", "mp4", "avi", "mkv", "MOV", "MP4", "AVI", "MKV"],
        help="支持 MOV、MP4、AVI、MKV 等格式的视频"
    )
    
    if uploaded_file is not None:
        # 获取文件扩展名并转换为小写
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # 验证文件扩展名
        allowed_extensions = ['.mov', '.mp4', '.avi', '.mkv']
        if file_extension not in allowed_extensions:
            st.error(f"不支持的文件格式：{file_extension}。请上传 MOV、MP4、AVI 或 MKV 格式的视频。")
            st.stop()
        
        # 保存上传的视频到临时文件
        with NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        try:
            # 尝试预览视频
            video_bytes = uploaded_file.getvalue()
            st.sidebar.video(video_bytes)
            
            # 获取视频信息
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("无法打开视频文件，请确保视频格式正确")
                os.unlink(video_path)
                st.stop()
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps
            cap.release()
            
            # 显示视频信息
            st.sidebar.text(f"视频信息:")
            st.sidebar.text(f"总帧数: {total_frames}")
            st.sidebar.text(f"原始帧率: {fps} FPS")
            st.sidebar.text(f"时长: {duration:.1f} 秒")
            
            if st.sidebar.button("开始处理"):
                with st.spinner("正在处理视频..."):
                    start_time = time.time()
                    st.session_state.video_frames = process_video(video_path)
                    st.session_state.processed_data = st.session_state.video_frames[0]  # 显示第一帧
                    st.session_state.current_frame = 0
                    process_time = time.time() - start_time
                    st.sidebar.success(f"处理完成！耗时 {process_time:.1f} 秒")
        
        except Exception as e:
            st.error(f"处理视频时出错: {str(e)}")
        finally:
            # 删除临时文件
            if os.path.exists(video_path):
                os.unlink(video_path)

# 自定义CSS
st.markdown("""
<style>
    /* 重置基础样式 */
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }

    /* 隐藏头部和页脚 */
    header[data-testid="stHeader"],
    footer {
        display: none !important;
    }

    /* 侧边栏样式 */
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

    /* 恢复侧边栏组件的样式 */
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

    /* 主内容区域样式 */
    .main {
        position: absolute !important;
        left: 350px !important;
        right: 0 !important;
        top: 0 !important;
        bottom: 0 !important;
        background: transparent !important;
    }

    /* Desmos容器样式 */
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

    /* 移除滚动条 */
    ::-webkit-scrollbar {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# 如果已经处理了图片或视频，则显示结果
if st.session_state.processed_data is not None:
    # 生成Desmos嵌入代码
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

        // 设置表达式面板位置
        calculator.updateSettings({{
            expressionsOrigin: {{ x: "left" }}
        }});
        
        // 设置视图范围
        calculator.setMathBounds({{
            left: -10,
            right: 10,
            bottom: -10,
            top: 10
        }});
        
        // 监听窗口大小变化
        window.addEventListener('resize', function() {{
            // 保持视图范围不变，让Desmos自动处理缩放
            calculator.setMathBounds({{
                left: -10,
                right: 10,
                bottom: -10,
                top: 10
            }});
        }});
        
        var currentFrame = 0;
        var isPlaying = false;
        var frameInterval = {FRAME_INTERVAL if MEDIA_TYPE == "视频" else 0};
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
                // 视频模式
                isPlaying = true;
                playAnimation();
            }} else {{
                // 图片模式
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
                                console.log("全部加载完成: " + exprs.length + " 个表达式");
                            }}
                        }}
                    }}
                    loadBatch();
                }}
                
                loadExpressionsInBatches(expressions, 100, 100);
            }}
            
        }} catch(e) {{
            console.error("加载表达式出错:", e);
        }}
    </script>
    """
    st.components.v1.html(desmos_html, height=1000, scrolling=False)
