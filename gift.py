import os
import sys
import subprocess
import yaml
import shutil
from pathlib import Path

# --- Configuration ---
CONFIG_FILE = "config.yml"
CAPTURE_SCRIPT = os.path.join("tools", "capture.py")
ANALYZE_SCRIPT = os.path.join("tools", "analyze.py")

# --- Main Logic ---

def check_file_exists(filename):
    """Checks if a required script file exists."""
    if not Path(filename).is_file():
        print(f"❌ 错误: 必需的脚本 '{filename}' 未找到。")
        print(f"   请确保 '{filename}' 与 gift.py 位于同一目录下。")
        sys.exit(1)

def load_config():
    """Loads student ID and password from the YAML config file."""
    check_file_exists(CONFIG_FILE)
    print(f"⚙️ 正在从 '{CONFIG_FILE}' 读取配置...")
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        student_id = config.get('stu_id')
        password = config.get('stu_pwd')

        if not student_id or not password:
            raise ValueError("'stu_id' 或 'stu_pwd' 字段缺失或为空。")
        
        print(f"   - 学号识别成功: {student_id}")
        return str(student_id), str(password)

    except (yaml.YAMLError, ValueError, TypeError) as e:
        print(f"❌ 错误: '{CONFIG_FILE}' 文件格式不正确或内容缺失。")
        print(f"   - 详情: {e}")
        print(f"   - 请确保文件包含有效的 'stu_id' 和 'stu_pwd'。")
        sys.exit(1)

def run_capture(student_id, password):
    """Runs the data capture script as a subprocess."""
    check_file_exists(CAPTURE_SCRIPT)
    print("\n" + "="*50)
    print("🚀 步骤 1/2: 开始自动捕获课程数据...")
    print("   这可能需要几分钟时间，请耐心等待...")
    print("="*50)

    if os.path.exists("tmp.json"):
        print("检测到已有tmp.json文件，跳过捕获数据阶段")
        return

    command = [sys.executable, CAPTURE_SCRIPT, student_id, password]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True)
        try:
            # 首先尝试使用 UTF-8 解码
            stdout_str = result.stdout.decode('utf-8')
            stderr_str = result.stderr.decode('utf-8')
        except UnicodeDecodeError:
            # 如果 UTF-8 解码失败，就回退到 GBK 编码。
            # 这在中文 Windows 环境下很常见。
            print("   - 信息: 捕获脚本的输出不是UTF-8编码，正在尝试使用GBK解码...")
            stdout_str = result.stdout.decode('gbk', errors='replace') # 'replace'可以防止因个别无法解码的字符而报错
            stderr_str = result.stderr.decode('gbk', errors='replace')

        print("✅ 数据捕获成功！")
        # 如果需要查看子脚本的详细输出，可以取消下面这行的注释
        # print(stdout_str)

    except subprocess.CalledProcessError as e:
        print("❌ 错误: 数据捕获脚本执行失败。")
        print("   - 请检查您的学号、密码以及网络连接。")
        print("\n--- 捕获脚本输出的错误信息 ---")
        try:
            stdout_err = e.stdout.decode('gbk', errors='replace')
            stderr_err = e.stderr.decode('gbk', errors='replace')
            print(stdout_err)
            print(stderr_err)
        except Exception:
            print(e.stdout)
            print(e.stderr)
        print("---------------------------\n")
        sys.exit(1)

def run_analysis(student_id):
    """Prepares files and runs the analysis script."""
    check_file_exists(ANALYZE_SCRIPT)
    
    # The analysis script expects 'result1.txt' and 'config.yml'
    # The capture script creates 'result_<id>.json'
    # We need to link them.
    
    source_data_file = Path(f"tmp.json")
    if not source_data_file.exists():
        print(f"❌ 错误: 未找到捕获的数据文件 '{source_data_file}'。")
        print("   捕获步骤可能没有成功生成数据。")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("📊 步骤 2/2: 开始生成您的个性化分析报告...")
    print("="*50)
    
    command = [sys.executable, ANALYZE_SCRIPT]

    try:
        # Run the analysis script. It will print its output directly.
        subprocess.run(command, check=True)
        print("\n🎉 恭喜！您的个人OO学习报告已生成完毕！")
    except subprocess.CalledProcessError:
        print(f"❌ 错误: 分析脚本 '{ANALYZE_SCRIPT}' 执行失败。")
    finally:
        os.remove(source_data_file)
        pass


if __name__ == "__main__":
    student_id, password = load_config()
    run_capture(student_id, password)
    run_analysis(student_id)