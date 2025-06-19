import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sys
import traceback
import dateutil
import numpy as np
import pandas as pd

CONFIG = {
    "FILE_PATH": os.path.join(".cache", "tmp.json"),
    "USER_INFO": {
        "real_name": None, "name": None, "student_id": None, "email": None
    },
    "HOMEWORK_NUM_MAP": {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8,
        '九': 9, '十': 10, '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15
    },
}

def exit_with_error(message: str):
    """打印一条致命错误信息到 stderr 并以状态码 1 退出脚本。"""
    print(f"\n[CRITICAL ERROR] {message}", file=sys.stderr)
    print("[INFO] Script terminated due to a critical error.", file=sys.stderr)
    sys.exit(1)

def get_hw_number(hw_name, config):
    match = re.search(r'第(.*)次作业', hw_name or '')
    return config["HOMEWORK_NUM_MAP"].get(match.group(1), 99) if match else 99

def is_target_user(data_dict, config):
    if not isinstance(data_dict, dict): return False
    user_id = str(config["USER_INFO"]["student_id"])
    if 'student_id' in data_dict and str(data_dict['student_id']) == user_id:
        return True
    return any(data_dict.get(k) == v for k, v in config["USER_INFO"].items() if v is not None)

def update_user_info(student_id, raw_data, config):
    user_name = None
    user_email = None
    try:
        for item in raw_data:
            if not isinstance(item, dict): continue
            body_data = item.get("body", {}).get("data", {})
            if not body_data: continue
            if 'mutual_test/room/self' in item.get('url', ''):
                for member in body_data.get('members', []):
                    if member.get('student_id') == student_id:
                        user_name = member.get('real_name')
                        break
            if user_name: break

        if user_name:
            for item in raw_data:
                if not isinstance(item, dict): continue
                body_data = item.get("body", {}).get("data", {})
                if not body_data: continue
                if 'ultimate_test/submit' in item.get('url', ''):
                    user_obj = body_data.get('user', {})
                    if user_obj.get('name') == user_name:
                        user_email = user_obj.get('email')
                        break
    except Exception as e:
        exit_with_error(f"An unexpected error occurred while searching for user info: {e}")

    if not user_name:
        exit_with_error(f"Could not find user with student ID '{student_id}' in the data file '{config['FILE_PATH']}'.")

    if not user_email:
        user_email = f"{student_id}@buaa.edu.cn"
        print(f"[WARNING] Could not find user email, auto-generating: {user_email}")

    config["USER_INFO"].update({
        "student_id": student_id, "real_name": user_name,
        "name": user_name, "email": user_email
    })
    print(f"Successfully identified user: {user_name} ({student_id})")
    return True


def parse_course_data(raw_data, config):
    homework_data = {}
    for item in raw_data:
        try:
            if not isinstance(item, dict): continue
            url = item.get('url', '')
            match = re.search(r'/homework/(\d+)', url)
            if not match: continue
            hw_id = match.group(1)
            if hw_id not in homework_data: homework_data[hw_id] = {'id': hw_id}

            body_data = item.get("body", {}).get("data", {})
            if not body_data: continue

            # 1. 作业元数据 (原始)
            if 'homework' in body_data:
                homework_data[hw_id].update(body_data['homework'])
                if 'has_mutual_test' not in homework_data[hw_id]:
                    homework_data[hw_id]['has_mutual_test'] = False

            # 2. 时间与阶段数据 (原始)
            if 'public_test' in url and 'public_test' in body_data:
                pt_data = body_data['public_test']
                homework_data[hw_id].update({
                    'public_test_used_times': pt_data.get('used_times'),
                    'public_test_start_time': pt_data.get('start_time'),
                    'public_test_end_time': pt_data.get('end_time'),
                    'public_test_last_submit': pt_data.get('last_submit'),
                })
            elif 'mutual_test' in url and 'room' not in url and 'data_config' not in url and 'start_time' in body_data:
                homework_data[hw_id].update({
                    'mutual_test_start_time': body_data.get('start_time'),
                    'mutual_test_end_time': body_data.get('end_time'),
                    'has_mutual_test': True
                })

            # 3. 个人表现与系统记录 (原始)
            elif 'ultimate_test/submit' in url and is_target_user(body_data.get('user', {}), config):
                homework_data[hw_id]['strong_test_score'] = body_data.get('score')
                homework_data[hw_id]['strong_test_details'] = body_data.get('results', [])
                if 'style' in body_data and 'score' in body_data.get('style', {}):
                    homework_data[hw_id]['style_score'] = body_data['style']['score']
                if 'uml_results' in body_data and body_data['uml_results']:
                    homework_data[hw_id]['uml_detailed_results'] = body_data['uml_results']

            # 4. 互测原始日志 (核心原始数据)
            elif 'mutual_test/room/self' in url:
                homework_data[hw_id]['has_mutual_test'] = True
                homework_data[hw_id]['room_members'] = body_data.get('members', []) # <-- 保留完整的成员列表
                homework_data[hw_id]['room_events'] = body_data.get('events', [])   # <-- 保留完整的事件日志
                # 从成员列表中找到自己，获取别名和房间等级
                for member in body_data.get('members', []):
                    if is_target_user(member, config):
                        homework_data[hw_id]['alias_name'] = member.get('alias_name_string')
                        homework_data[hw_id]['room_level'] = body_data.get('mutual_test', {}).get('level', 'N/A').upper()
                        break

            # 5. Bug修复原始数据
            elif 'bug_fix' in url and 'personal' in body_data:
                homework_data[hw_id]['bug_fix_details'] = body_data.get('personal', {}) # <-- 保留完整的 personal 字典
        
        except (KeyError, TypeError, AttributeError) as e:
            url_info = item.get('url', 'N/A') if isinstance(item, dict) else 'Unknown Item'
            print(f"[WARNING] Skipping a malformed data item. URL: {url_info}. Error: {e}", file=sys.stderr)
            continue
            
    # processed_homeworks 将只包含纯粹的作业数据
    processed_homeworks = [data for hw_id, data in homework_data.items() if 'name' in data]
    
    def get_sort_key(hw_data):
        return get_hw_number(hw_data.get('name', ''), config)
        
    return sorted(processed_homeworks, key=get_sort_key)

# 在 preprocess.py 中

def parse_forum_data(raw_data, config, df):
    user_name = config["USER_INFO"]["real_name"]
    hw_forum_activities = {hw_num: [] for hw_num in df['hw_num'].unique()}

    for item in raw_data:
        try:
            if not isinstance(item, dict) or '/post/' not in item.get('url', ''):
                continue
            
            post_data = item.get("body", {}).get("data", {})
            post = post_data.get('post')
            if not post or not post_data.get('homework'):
                continue
            
            hw_name = post_data['homework']['name']
            hw_num = get_hw_number(hw_name, config)
            if hw_num not in hw_forum_activities:
                continue

            # 检查发帖行为 (authored)
            if post.get('user_name') == user_name:
                activity = {
                    'type': 'authored',
                    'title': post.get('title', '无标题'),
                    'category': post.get('category'), # issue, free_discuss
                    'priority': post.get('priority')  # top, essential, normal
                }
                hw_forum_activities[hw_num].append(activity)

            # 检查回帖行为 (commented)
            post_author = post.get('user_name')
            post_category = post.get('category')
            post_priority = post.get('priority')
            comments = post_data.get('comments', [])
            for comment in comments:
                if comment.get('user_name') == user_name:
                    activity = {
                        'type': 'commented',
                        'post_title': post.get('title', '无标题'),
                        'post_author': post_author,
                        'post_category': post_category,
                        'post_priority': post_priority
                    }
                    hw_forum_activities[hw_num].append(activity)
        
        except (AttributeError, TypeError, KeyError) as e:
            # ... (错误处理) ...
            exit_with_error(f"遇到某些难以解决的问题: {e}")

    # 将这个原始活动列表作为一个新列添加到 DataFrame
    df['forum_activities'] = df['hw_num'].map(hw_forum_activities)
    
    return df

def parse_commit_data(raw_data, config):
    commit_history = {}
    pattern = re.compile(r'(\d+)月\s*(\d+),\s*(\d{4})\s*(\d{1,2}):(\d{2})(下午|上午)\s*GMT\+0800')

    for item in raw_data:
        try:
            if not isinstance(item, dict): continue
            if "hw" in item and "commits" in item:
                hw_num = int(item['hw'])
                commits = item.get('commits', {})
                if not isinstance(commits, dict): continue
                
                commit_list = []
                for timestamp_str, message in commits.items():
                    try:
                        match = pattern.match(timestamp_str)
                        if not match:
                            processed_ts = timestamp_str.replace('月 ', '/').replace(',', '').replace('下午', 'PM').replace('上午', 'AM')
                            timestamp_aware = dateutil.parser.parse(processed_ts)
                        else:
                            month, day, year, hour, minute, am_pm = match.groups()
                            hour, minute, day, month, year = map(int, [hour, minute, day, month, year])
                            if am_pm == '下午' and hour != 12: hour += 12
                            elif am_pm == '上午' and hour == 12: hour = 0
                            beijing_tz = timezone(timedelta(hours=8))
                            timestamp_aware = datetime(year, month, day, hour, minute, tzinfo=beijing_tz)
                        
                        timestamp_utc = timestamp_aware.astimezone(timezone.utc)
                        timestamp_naive_utc = timestamp_utc.replace(tzinfo=None)
                        commit_list.append({'timestamp': timestamp_naive_utc, 'message': message})
                    except (dateutil.parser.ParserError, TypeError, ValueError) as e:
                        print(f"[WARNING] Could not parse a commit timestamp for HW {hw_num}: '{timestamp_str}'. Skipping. Error: {e}", file=sys.stderr)
                        continue
                if commit_list:
                    commit_history[hw_num] = sorted(commit_list, key=lambda x: x['timestamp'])
        except (ValueError, TypeError) as e:
            print(f"[WARNING] Skipping a malformed commit data item. Item: {item}. Error: {e}", file=sys.stderr)
            continue
    return commit_history

def filter_timeline(df):
    dt_cols = ['public_test_start_time', 'public_test_end_time']
    for col in dt_cols:
        if col in df.columns:
            # errors='coerce' 会将无法解析的日期变为 NaT (Not a Time)
            # utc=True, tz_localize(None) 是为了和 commit 时间戳的格式保持一致（无时区信息的UTC时间）
            df[col] = pd.to_datetime(df[col], errors='coerce') \
            .dt.tz_localize('Asia/Shanghai') \
            .dt.tz_convert('UTC')
            df[col] = df[col].dt.tz_localize(None)

    # 定义一个内部函数，用于处理 DataFrame 的每一行
    def filter_row_commits(row):
        # print(row.dtype)
        start_time = row['public_test_start_time']
        end_time = row['public_test_end_time']
        commits = row['commits']

        # 如果起止时间无效或 commit 列表为空，则直接返回空列表，避免错误
        if pd.isna(start_time) or pd.isna(end_time) or not commits:
            return []
        release_time = start_time - timedelta(days=3)
        filtered_commits = [
            c for c in commits 
            if release_time <= c['timestamp'] <= end_time
        ]
        # pprint(filtered_commits)
        
        return filtered_commits
    df['commits'] = df.apply(filter_row_commits, axis=1)
    return df

def main(student_id):
    try:
        try:
            file_path = Path(CONFIG["FILE_PATH"])
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            exit_with_error(f"数据文件 '{CONFIG['FILE_PATH']}' 未找到。请先运行 capture.py。")
        except json.JSONDecodeError:
            exit_with_error(f"数据文件 '{CONFIG['FILE_PATH']}' 不是一个有效的JSON文件，可能已损坏。")

        update_user_info(student_id, raw_data, CONFIG)

        homework_details = parse_course_data(raw_data, CONFIG)
        if not homework_details:
            exit_with_error("未找到该学生的任何有效作业数据，无法生成报告。")
        commit_history = parse_commit_data(raw_data, CONFIG)
        raw_df = pd.DataFrame(homework_details)
        # some details
        raw_df['hw_num'] = raw_df['name'].apply(lambda x: get_hw_number(x, CONFIG))
        raw_df['commits'] = raw_df['hw_num'].map(commit_history).fillna('').apply(list)
        print(raw_df.axes[1])
        # raw_df.to_csv("tmp.csv")
        df_filtered = filter_timeline(raw_df)
        # print(df_filtered.axes[1])
        df:pd.DataFrame = parse_forum_data(raw_data, CONFIG, df_filtered)
        print(df.axes[1])
        # print(df_metrics.axes[1])
        # print(df.axes[1])
        # print(set(df.axes[1]).difference(df_metrics.axes[1]))
        if 'has_self_test' in df.columns:
            df = df.drop(columns=['has_self_test'])
            print("[INFO] 'has_self_test' column has been removed.")
        df.to_pickle(os.path.join(".cache", "tmp.pkl"))
        # df.to_csv("tmp.csv")
        json.dump(CONFIG["USER_INFO"], open(os.path.join(".cache", "user.info"), "w", encoding="utf-8"))
    except SystemExit:
        pass
    except Exception as e:
        print("\n" + "="*80, file=sys.stderr)
        print(" AN UNEXPECTED ERROR OCCURRED ".center(80, "#"), file=sys.stderr)
        print("="*80, file=sys.stderr)
        print("脚本在处理过程中遇到一个未知错误，这可能是由数据格式问题或代码逻辑缺陷引起的。", file=sys.stderr)
        print("请检查数据文件是否完整，或将以下错误信息报告给开发者。", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # main("23371265")
    parser = argparse.ArgumentParser(description="Capture API data from BUAA OO course website.")
    parser.add_argument("student_id", help="Your student ID (e.g., 23371265)")
    args = parser.parse_args()
    main(args.student_id)