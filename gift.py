# --- START OF FILE gift.py ---

import os
import sys
import subprocess
import yaml
from pathlib import Path
import io

# --- Configuration ---
CONFIG_FILE = "config.yml"
CAPTURE_SCRIPT = os.path.join("tools", "capture.py")
ANALYZE_SCRIPT = os.path.join("tools", "analyze.py")
DATA_FILE = "tmp.json"
CLEANUP = False
DEBUG_MODE = False

# --- Helper Functions ---

def check_file_exists(filename, is_critical=True):
    if not Path(filename).is_file():
        if is_critical:
            print(f"❌ 致命错误: 必需的文件 '{filename}' 未找到。")
            print(f"   请确保此文件存在于正确的位置。")
            sys.exit(1)
        return False
    return True

def read_and_print_stream_live(stream_name, byte_stream, show_output):
    try:
        first_line_bytes = byte_stream.readline()
    except Exception:
        return # 如果流读取出错，直接返回

    if not first_line_bytes:
        return # 如果流中没有任何数据，直接返回

    determined_encoding = 'utf-8'
    try:
        first_line_str = first_line_bytes.decode('utf-8')
    except UnicodeDecodeError:
        determined_encoding = 'gbk'
        first_line_str = first_line_bytes.decode('gbk', errors='replace')
        if show_output: # 仅在需要显示时打印警告
            print(f"[解码警告] {stream_name} 的编码不是标准的 UTF-8，已尝试回退到 GBK。", file=sys.stderr)
    if show_output:
        print(first_line_str.strip())
    with io.TextIOWrapper(byte_stream, encoding=determined_encoding, errors='replace') as text_stream:
        for line in text_stream:
            if show_output:
                print(line.strip())

def run_subprocess_live(command, description, show_stdout=True):
    print("\n" + "="*50)
    print(f"🚀 正在执行: {description}...")
    if show_stdout:
        print("   (您将看到来自子脚本的实时输出)")
    else:
        print("   (调试模式已关闭，将隐藏过程日志，请耐心等待...)")
    print("="*50)
    my_env = os.environ.copy()
    my_env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=my_env
    )

    output_buffer = []

    try:
        stdout_reader = io.TextIOWrapper(process.stdout, encoding='utf-8', errors='replace')
        stderr_reader = io.TextIOWrapper(process.stderr, encoding='gbk', errors='replace')

        while True:
            line = stdout_reader.readline()
            if not line:
                break
            if show_stdout:
                print(line.strip())
            output_buffer.append(line)
        
        process.wait()
        stderr_output = stderr_reader.read()
        if stderr_output:
            print("\n--- 脚本的错误输出流 ---", file=sys.stderr)
            print(stderr_output, file=sys.stderr)
            print("------------------------\n", file=sys.stderr)
            output_buffer.append(stderr_output)

    except Exception as e:
        print(f"❌ 运行子进程时发生内部错误: {e}")
        process.kill()
        sys.exit(1)


    if process.returncode != 0:
        print(f"❌ 错误: '{description}' 步骤执行失败。")
        print(f"   子脚本返回了错误码: {process.returncode}")
        print("   请检查上面由子脚本打印的具体错误信息。")
        sys.exit(1)

    print(f"✅ '{description}' 步骤成功完成！")


def load_config():
    check_file_exists(CONFIG_FILE)
    print(f"⚙️  正在从 '{CONFIG_FILE}' 读取配置...")
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        student_id = config.get('stu_id')
        password = config.get('stu_pwd')
        if not student_id or not password:
            raise ValueError("'stu_id' 或 'stu_pwd' 字段缺失或为空。")
        
        cleanup = config.get('cleanup')
        if cleanup is None or not isinstance(cleanup, bool):
            raise ValueError("'cleanup' 字段缺失或类型不是布尔值 (true/false)。")
        
        debug = config.get('debug')
        if debug is None or not isinstance(debug, bool):
            raise ValueError("'debug' 字段缺失或类型不是布尔值 (true/false)。")

        global CLEANUP, DEBUG_MODE
        CLEANUP = cleanup
        DEBUG_MODE = debug
        
        print(f"   - 学号识别成功: {student_id}")
        print(f"   - 清理模式: {'开启' if CLEANUP else '关闭'}")
        print(f"   - 调试模式: {'开启' if DEBUG_MODE else '关闭'}")
        return str(student_id), str(password)

    except (yaml.YAMLError, ValueError, TypeError) as e:
        print(f"❌ 错误: '{CONFIG_FILE}' 文件格式不正确或内容缺失。")
        print(f"   - 详情: {e}")
        print(f"   - 请确保文件包含有效的 'stu_id', 'stu_pwd', 'cleanup', 'debug' 字段。")
        sys.exit(1)

def run_capture(student_id, password):
    check_file_exists(CAPTURE_SCRIPT)
    if Path(DATA_FILE).exists() and not DEBUG_MODE:
        print(f"✨ 检测到已存在的数据文件 '{DATA_FILE}'。")
        choice = input("   您想跳过数据捕获，直接进行分析吗？(y/n): ").lower()
        if choice == 'y':
            print("   好的，已跳过数据捕割步骤。")
            return
        else:
            print("   好的，将重新捕获数据...")

    command = [sys.executable, CAPTURE_SCRIPT, student_id, password, str(DEBUG_MODE)]
    run_subprocess_live(command, "步骤 1/2: 数据捕获", show_stdout=DEBUG_MODE)

def run_analysis():
    check_file_exists(ANALYZE_SCRIPT)
    check_file_exists(DATA_FILE)
    check_file_exists(os.path.join("tools", "corpus.json"))
    
    command = [sys.executable, ANALYZE_SCRIPT]
    run_subprocess_live(command, "步骤 2/2: 报告分析与生成", show_stdout=True)

    try:
        if CLEANUP:
            print(f"\n✨ 正在清理临时数据文件 '{DATA_FILE}'...")
            os.remove(DATA_FILE)
            print("   清理完成。")
    except OSError as e:
        print(f"   [警告] 清理临时文件失败: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print(" BUAA OO 课程学习轨迹分析报告生成器 ".center(54))
    print("=" * 60)
    
    student_id, password = load_config()
    run_capture(student_id, password)
    run_analysis()
    
    print("\n" + "*"*60)
    print("🎉 恭喜！所有步骤已成功完成！".center(54))
    print("   您的个人OO学习报告和图表已生成完毕！".center(52))
    print("*"*60)