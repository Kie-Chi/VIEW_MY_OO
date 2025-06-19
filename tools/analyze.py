
from datetime import timedelta, datetime, timezone
import json
import os
import pprint
import re
import dateutil.parser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from collections import Counter
import sys
import traceback
import yaml

"""
动态个性化面向对象课程数据分析脚本 V8.7 (Refactored Corpus & Hardened)

此版本保留了所有原始分析逻辑，但增强了错误处理能力。
- 对于关键性错误（如文件丢失、用户未找到），脚本将打印清晰信息并正常退出。
- 对于非关键性错误（如单个数据点格式错误），脚本将打印警告并继续分析，以生成尽可能完整的报告。
"""

# --- 1. 配置区 ---
CONFIG = {
    "FILE_PATH": os.path.join(".cache", "tmp.pkl"),
    "USER_PATH": os.path.join(".cache", "user.info"),
    "YAML_CONFIG_PATH": "config.yml",
    "USER_INFO": {
        "real_name": None, "name": None, "student_id": None, "email": None
    },
    "UNIT_MAP": {
        "第一单元：表达式求导": list(range(1, 5)),
        "第二单元：多线程电梯": list(range(5, 9)),
        "第三单元：JML规格化设计": list(range(9, 13)),
        "第四单元：UML解析": list(range(13, 16)),
    },
}
# --- Matplotlib 设置 ---
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 语料库 ---
REPORT_CORPUS = {}

# --- [新增] 错误处理函数 ---
def exit_with_error(message: str):
    """打印一条致命错误信息到 stderr 并以状态码 1 退出脚本。"""
    print(f"\n[CRITICAL ERROR] {message}", file=sys.stderr)
    print("[INFO] Script terminated due to a critical error.", file=sys.stderr)
    sys.exit(1)



def is_target_user(data_dict, config):
    if not isinstance(data_dict, dict): return False
    user_id = str(config["USER_INFO"]["student_id"])
    if 'student_id' in data_dict and str(data_dict['student_id']) == user_id:
        return True
    return any(data_dict.get(k) == v for k, v in config["USER_INFO"].items() if v is not None)

def hw2unit(hw_num, config):
    for unit_name, hw_nums in config["UNIT_MAP"].items():
        if hw_num in hw_nums:
            return unit_name
    return "其他"

def enrich(df, config):
    
    # --- 步骤 0: 确保基础列存在且类型正确 (来自旧的 calculate_derived_metrics) ---
    # 这步非常重要，为后续所有计算提供保障
    dict_cols = ['bug_fix_details']
    for col in dict_cols:
        if col in df.columns:
            # 这里的 bug_fix_details 可能是从JSON来的 dict，也可能是NaN
            df[col] = df[col].apply(lambda x: x if isinstance(x, dict) else {})
        else:
            df[col] = [{} for _ in range(len(df))]
            
    list_cols = ['strong_test_details', 'room_members', 'room_events', 'commits', 'uml_detailed_results']
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        else:
            df[col] = [[] for _ in range(len(df))]

    # --- 步骤 1: 计算第一层衍生指标 (来自旧的 calculate_derived_metrics) ---
    df['unit'] = df['hw_num'].apply(lambda x: hw2unit(x, config))

    # 强测相关衍生
    df['strong_test_issues'] = df['strong_test_details'].apply(
        lambda details: dict(Counter(p['message'] for p in details if p.get('message') != 'ACCEPTED'))
    )

    # 互测相关衍生
    df['room_member_count'] = df['room_members'].apply(len)
    df['room_total_hacked'] = df['room_members'].apply(
        lambda members: sum(int(m.get('hacked', {}).get('success', 0)) for m in members)
    )
    df['room_total_hack_success'] = df['room_members'].apply(
        lambda members: sum(int(m.get('hack', {}).get('success', 0)) for m in members)
    )
    df['room_total_hack_attempts'] = df['room_members'].apply(
        lambda members: sum(int(m.get('hack', {}).get('total', 0)) for m in members)
    )

    # 个人互测统计
    def get_my_mutual_stats(row):
        # is_target_user 函数需要可用
        stats = {'hack_success': 0, 'hack_total_attempts': 0, 'hacked_success': 0, 'hacked_total_attempts': 0}
        for member in row['room_members']:
            if is_target_user(member, config):
                stats['hack_success'] = int(member.get('hack', {}).get('success', 0))
                stats['hack_total_attempts'] = int(member.get('hack', {}).get('total', 0))
                stats['hacked_success'] = int(member.get('hacked', {}).get('success', 0))
                stats['hacked_total_attempts'] = int(member.get('hacked', {}).get('total', 0))
                break
        return pd.Series(stats)
    df[['hack_success', 'hack_total_attempts', 'hacked_success', 'hacked_total_attempts']] = df.apply(get_my_mutual_stats, axis=1)

    # 个人成功 hack 事件列表
    df['mutual_test_events'] = df['room_events'].apply(
        lambda events: [e for e in events if is_target_user(e.get('hack', {}), config)]
    )
    
    # Bug修复相关衍生
    # 注意：原始 bug_fix_details 是 {'hack': {...}, 'hacked': {...}} 结构
    df['bug_fix_hack_score'] = df['bug_fix_details'].apply(lambda x: x.get('hack', {}).get('score', 0))
    df['bug_fix_hacked_score'] = df['bug_fix_details'].apply(lambda x: x.get('hacked', {}).get('score', 0))
    df['bug_fix_hacked_count'] = df['bug_fix_details'].apply(lambda x: x.get('hacked', {}).get('count', 0))
    df['bug_fix_unfixed_count'] = df['bug_fix_details'].apply(lambda x: x.get('hacked', {}).get('unfixed', 0))

    # --- 步骤 2: 计算第二层衍生指标和进行数据类型转换 (来自旧的 preprocess_and_calculate_metrics) ---
    
    # 时间列处理
    dt_cols = ['public_test_start_time', 'public_test_end_time', 'public_test_last_submit',
               'mutual_test_start_time', 'mutual_test_end_time']
    for col in dt_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.tz_localize(None)

    # 基于时间的衍生指标
    durations = (df['public_test_end_time'] - df['public_test_start_time']).dt.total_seconds()
    offsets = (df['public_test_last_submit'] - df['public_test_start_time']).dt.total_seconds()
    df['ddl_index'] = (offsets / durations).fillna(0.5).clip(0, 1)

    # 基于互测统计的衍生指标
    df['offense_defense_ratio'] = (df['hack_success'].fillna(0) + 1) / (df['hacked_success'].fillna(0) + 1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        df['hack_success_rate'] = (df['hack_success'] / df['hack_total_attempts']) * 100
    df['hack_success_rate'] = df['hack_success_rate'].fillna(np.nan)

    # 基于强测和互测的衍生指标
    df['strong_test_deduction_count'] = df['strong_test_issues'].apply(lambda x: sum(x.values()))
    room_weights = {'A': 10, 'B': 8, 'C': 5}
    df['weighted_defense_deduction'] = df.apply(
        lambda row: row.get('hacked_success', 0) * room_weights.get(row.get('room_level'), 3), axis=1)

    # 基于Bug修复的衍生指标
    with np.errstate(divide='ignore', invalid='ignore'):
        df['bug_fix_rate'] = ((df['bug_fix_hacked_count'] - df['bug_fix_unfixed_count']) / df['bug_fix_hacked_count']) * 100
    df['bug_fix_rate'] = df['bug_fix_rate'].fillna(np.nan)
    df['hack_fix_score_ratio'] = (df['bug_fix_hack_score'] + 0.1) / (df['bug_fix_hacked_score'] + 0.1)

    # --- 步骤 3: 基于 commits 的深度衍生指标 (开发过程分析) ---

    df['commit_count'] = df['commits'].apply(len)
    df['work_start_time'] = pd.to_datetime(df['commits'].apply(lambda x: x[0]['timestamp'] if x else pd.NaT), utc=True).dt.tz_localize(None)
    df['work_end_time'] = pd.to_datetime(df['commits'].apply(lambda x: x[-1]['timestamp'] if x else pd.NaT), utc=True).dt.tz_localize(None)
    df['work_span_hours'] = (df['work_end_time'] - df['work_start_time']).dt.total_seconds() / 3600
    
    total_public_duration = (df['public_test_end_time'] - df['public_test_start_time']).dt.total_seconds()
    work_start_offset = (df['work_start_time'] - df['public_test_start_time']).dt.total_seconds()
    df['start_ratio'] = (work_start_offset / total_public_duration).fillna(1.0).clip(0, 1)

    def calculate_cadence(commits):
        if len(commits) < 3: return 0
        timestamps = [c['timestamp'].timestamp() for c in commits]
        return np.std(timestamps) / 3600
    df['work_cadence_std_dev'] = df['commits'].apply(calculate_cadence)
    
    def analyze_commit_messages(commits):
        if not commits: return {}
        version_pattern = re.compile(r'\b(?:v|V)-?(\d+(\.\d+)*)\b')
        counts = Counter()
        for commit in commits:
            msg_lower = commit.get('message', '').lower()
            if version_pattern.search(commit.get('message', '')): counts['versioning'] += 1
            if 'fix' in msg_lower or 'bug' in msg_lower or '修复' in msg_lower: counts['fix'] += 1
            if 'refactor' in msg_lower or 'rebuild' in msg_lower or '重构' in msg_lower: counts['refactor'] += 1
        return dict(counts)
    df['commit_keywords'] = df['commits'].apply(analyze_commit_messages)
    df['commit_refactor_count'] = df['commit_keywords'].apply(lambda x: x.get('refactor', 0))

    def get_development_style_tags(row):
        tags = []
        if row['commit_count'] < 1: return tags
        if row['start_ratio'] < 0.2: tags.append('EARLY_BIRD')
        elif row['work_cadence_std_dev'] > 12 and row['work_span_hours'] > 24: tags.append('WELL_PACED')
        else: tags.append('DDL_FIGHTER')
        if row['commit_refactor_count'] > 0: tags.append('PROCESS_REFACTORING')
        return tags
    df['dev_style_tags'] = df.apply(get_development_style_tags, axis=1)
    
    # --- 步骤 4: 清理不再需要的原始数据列 ---
    # 清理掉 room_members，因为它已经被分解为各个统计列
    df = df.drop(columns=['room_members'], errors='ignore')

    return df
# --- 5. 可视化模块 (原逻辑不变) ---
def create_visualizations(df, user_name, config):
    print("\n正在生成可视化图表，请稍候...")
    try:
        create_performance_dashboard(df, user_name)
    except Exception as e:
        print(f"[WARNING] 生成“综合表现仪表盘”失败，已跳过。错误: {e}", file=sys.stderr)
    try:
        create_unit_radar_chart(df, user_name, config)
    except Exception as e:
        print(f"[WARNING] 生成“各单元能力雷达图”失败，已跳过。错误: {e}", file=sys.stderr)
    print("所有分析报告与图表已生成完毕！")

def create_performance_dashboard(df, user_name):
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f'{user_name} - OO课程综合表现仪表盘 (V8.7)', fontsize=24, weight='bold')

    ax1 = axes[0, 0]
    df_strong = df.dropna(subset=['strong_test_score'])
    if not df_strong.empty:
        ax1.plot(df_strong['name'], df_strong['strong_test_score'], marker='o', linestyle='-', color='b', label='强测分数')
        ax1.axhline(y=100, color='r', linestyle='--', label='满分线 (100)', alpha=0.7)
        ax1.set_title('学期强测成绩变化趋势', fontsize=16)
        ax1.set_xlabel('作业', fontsize=12)
        ax1.set_ylabel('分数', fontsize=12)
        ax1.tick_params(axis='x', rotation=45, labelsize=9)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.legend()
        ax1.set_ylim(bottom=min(80, df_strong['strong_test_score'].min() - 5 if not df_strong.empty else 80), top=105)

    ax2 = axes[0, 1]
    analysis_df = df.dropna(subset=['ddl_index', 'strong_test_deduction_count', 'hacked_success'])
    if not analysis_df.empty:
        color1 = 'tab:red'
        ax2.set_xlabel('DDL 指数 (越接近1，越晚提交)', fontsize=12)
        ax2.set_ylabel('强测扣分点数量', color=color1, fontsize=12)
        ax2.scatter(analysis_df['ddl_index'], analysis_df['strong_test_deduction_count'],
                    alpha=0.6, color=color1, label='强测扣分点')
        ax2.tick_params(axis='y', labelcolor=color1)
        
        ax2_twin = ax2.twinx()
        color2 = 'tab:blue'
        ax2_twin.set_ylabel('被成功Hack次数', color=color2, fontsize=12)
        ax2_twin.scatter(analysis_df['ddl_index'], analysis_df['hacked_success'],
                         marker='x', alpha=0.6, color=color2, label='被成功Hack次数')
        ax2_twin.tick_params(axis='y', labelcolor=color2)
        ax2.set_title('提交时间与代码质量关联', fontsize=16)
    
    ax3 = axes[1, 0]
    mutual_df = df[df.get('has_mutual_test', pd.Series(False))].dropna(subset=['offense_defense_ratio'])
    if not mutual_df.empty:
        ax3.plot(mutual_df['name'], mutual_df['offense_defense_ratio'], marker='^', linestyle=':', color='purple', label='攻防指数')
        ax3.axhline(y=1, color='grey', linestyle='--', label='攻防平衡线 (指数=1)')
        ax3.set_yscale('log')
        ax3.set_title('互测攻防策略演化 (对数坐标)', fontsize=16)
        ax3.set_xlabel('作业', fontsize=12)
        ax3.set_ylabel('攻防指数 (Hack+1)/(Hacked+1)', fontsize=12)
        ax3.tick_params(axis='x', rotation=45, labelsize=9)
        ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax3.legend()

    ax4 = axes[1, 1]
    bugfix_df = df.dropna(subset=['bug_fix_rate'])
    if not bugfix_df.empty:
        bars = ax4.bar(bugfix_df['name'], bugfix_df['bug_fix_rate'], color='teal', alpha=0.8)
        ax4.axhline(y=100, color='green', linestyle='--', label='100%修复', alpha=0.7)
        ax4.set_title('Bug修复率', fontsize=16)
        ax4.set_xlabel('作业', fontsize=12)
        ax4.set_ylabel('修复率 (%)', fontsize=12)
        ax4.set_ylim(0, 110)
        ax4.tick_params(axis='x', rotation=45, labelsize=9)
        ax4.grid(axis='y', linestyle='--', linewidth=0.5)
        ax4.legend()
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                ax4.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.0f}%', va='bottom', ha='center')
    else:
        ax4.text(0.5, 0.5, '未发现可供分析的Bug修复数据', ha='center', va='center', fontsize=14, color='gray')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def create_unit_radar_chart(df, user_name, config):
    unit_stats = {}
    for unit_name in config["UNIT_MAP"].keys():
        unit_df = df[df['unit'] == unit_name]
        if not unit_df.empty:
            unit_stats[unit_name] = {
                '强测表现': unit_df['strong_test_score'].mean(skipna=True),
                '进攻能力': unit_df['hack_success'].sum(skipna=True),
                '防守能力': unit_df['hacked_success'].sum(skipna=True)
            }
    valid_units = {k: v for k, v in unit_stats.items() if pd.notna(v.get('强测表现'))}
    if len(valid_units) >= 3:
        labels = list(next(iter(valid_units.values())).keys())
        stats_list = [list(d.values()) for d in valid_units.values()]
        stats_array = np.array(stats_list)
        max_hacked = np.nanmax(stats_array[:, 2])
        if max_hacked > 0: stats_array[:, 2] = max_hacked - stats_array[:, 2] 
        else: stats_array[:, 2] = 1 # 如果从未被hack，给一个最高分
        
        with np.errstate(divide='ignore', invalid='ignore'):
            max_vals = np.nanmax(stats_array, axis=0)
            max_vals[max_vals == 0] = 1 # 避免除以0
            normalized_stats = stats_array / max_vals
        
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        for i, (unit_name, data) in enumerate(valid_units.items()):
            values = normalized_stats[i].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=re.sub(r'：.*', '', unit_name))
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title(f'{user_name} - 各单元能力雷达图', size=20, color='blue', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.show()

def analyze_submission_style(hw_row):
    start, end, last_submit = hw_row.get('public_test_start_time'), hw_row.get('public_test_end_time'), hw_row.get('public_test_last_submit')
    if pd.isna(start) or pd.isna(end) or pd.isna(last_submit): return random.choice(REPORT_CORPUS["SUBMISSION"]["STYLE"]["UNKNOWN"])
    total_duration = end - start
    if total_duration.total_seconds() <= 0: return random.choice(REPORT_CORPUS["SUBMISSION"]["STYLE"]["UNKNOWN"])
    ratio = (last_submit - start).total_seconds() / total_duration.total_seconds()
    if ratio <= 0.2: return random.choice(REPORT_CORPUS["SUBMISSION"]["STYLE"]["EARLY_BIRD"])
    elif ratio >= 0.8: return random.choice(REPORT_CORPUS["SUBMISSION"]["STYLE"]["DDL_FIGHTER"])
    else: return random.choice(REPORT_CORPUS["SUBMISSION"]["STYLE"]["WELL_PACED"])

def generate_highlights(df, config):
    """[V8.9-Upgraded] 生成个人亮点，计算元成就，并返回用于展示的列表和所有已解锁成就的详细信息。"""
    if df.empty:
        return [], {}

    # earned_achievements 将存储解锁的成就及其详细信息
    # 格式: {'KEY': {'description': '...', 'context': '于 ...'}}
    earned_achievements = {}
    df_with_mutual_opportunities = df[df['has_mutual_test'] == True]
    total_mutual_opportunities = len(df_with_mutual_opportunities)

    # 2. 筛选出学生实际进入了互测房间的作业 (用于计算分子)
    mutual_df = df_with_mutual_opportunities.dropna(subset=['room_level'])
    
    # 3. 计算比率，使用正确的分母
    a_rate, b_rate, c_rate = 0, 0, 0
    if total_mutual_opportunities > 0:
        # 分子：从实际进入的房间中统计 A、B、C 各有多少次
        room_counts = mutual_df['room_level'].value_counts()
        
        # 分母：使用总的互测机会次数
        a_rate = room_counts.get('A', 0) / total_mutual_opportunities
        b_rate = room_counts.get('B', 0) / total_mutual_opportunities
        c_rate = room_counts.get('C', 0) / total_mutual_opportunities

    total_hacks = mutual_df['hack_success'].sum()
    total_hacked = mutual_df['hacked_success'].sum()

    # 论坛活动统计
    total_forum_activity = df['essential_posts_authored'].sum() + df['official_replies'].sum() + df['peer_assists'].sum()
    
    # 3. [修改] 房间等级统计，使用正确的基数
    a_rate, b_rate, c_rate = 0, 0, 0
    if total_mutual_opportunities > 0:
        room_counts = mutual_df['room_level'].value_counts()
        a_rate = room_counts.get('A', 0) / total_mutual_opportunities # 使用总机会数作为分母
        b_rate = room_counts.get('B', 0) / total_mutual_opportunities # 使用总机会数作为分母
        c_rate = room_counts.get('C', 0) / total_mutual_opportunities # 使用总机会数作为分母

    def add_highlight(key, context_str, **kwargs):
        """辅助函数，添加成就并记录其描述和上下文。"""
        if key not in earned_achievements:
            template = REPORT_CORPUS["HIGHLIGHTS"]["TAGS"][key]["description"]
            earned_achievements[key] = {
                'description': template.format(**kwargs),
                'context': context_str
            }

    strong_scores = df['strong_test_score'].dropna()
    submit_times_df = df.dropna(subset=['public_test_used_times'])

    # --- 1. 基础成就检查 (为每个成就添加 context_str) ---
    # “千刀狼王”成就判断逻辑
    # 1. 计算整个学期的总Hack尝试次数和总成功次数
    total_hacks_attempted = df['hack_total_attempts'].sum()
    total_hacks_successful = df['hack_success'].sum()
    
    # 2. 避免除以零，并计算成功率
    if total_hacks_attempted > 0:
        success_rate = (total_hacks_successful / total_hacks_attempted) * 100
        
        # 3. 设置阈值并判断是否满足条件
        #    - 总攻击次数 > 200 (可自行调整)
        #    - 成功率 > 5% (可自行调整)
        if total_hacks_attempted > 500 and success_rate > 5:
            # 4. 如果满足条件，则添加这个成就
            add_highlight(
                "THOUSAND_BLADE_WOLF_KING",  # 使用在corpus.json中定义的Key
                "于 整个学期",             # 成就上下文
                total_attempts=int(total_hacks_attempted),
                total_hacks=int(total_hacks_successful),
                rate=success_rate
            )
    # 王座常客
    if total_mutual_opportunities > 0 and a_rate >= 0.75: # 使用修正后的 a_rate
        add_highlight("A_ROOM_REGULAR", "于 整个学期")

    # 玻璃大炮
    if total_hacks > 15 and total_hacked > 10:
        add_highlight("GLASS_CANNON", "于 整个学期")

    # 孤高剑客
    avg_score = strong_scores.mean() if not strong_scores.empty else 0
    if total_hacks <= 2 and total_hacked <= 2 and avg_score > 85:
        add_highlight("LONE_SWORDSMAN", "于 整个学期")

    # 过山车玩家
    if total_mutual_opportunities > 2 and max(a_rate, b_rate, c_rate) < 0.6:
        add_highlight("ROLLER_COASTER_RIDER", "于 整个学期")

    # 沉默是金
    if total_forum_activity <= 1:
        add_highlight("GOLDEN_SILENCE", "于 整个学期")
    # 社区之光
    essential_posts_df = df[df['essential_posts_authored'] > 0]
    if not essential_posts_df.empty:
        # 获取第一个精华帖的标题用于展示
        first_essential_title = essential_posts_df.iloc[0]['essential_post_titles'][0]
        add_highlight("COMMUNITY_PILLAR", f"于《{first_essential_title}》", post_title=first_essential_title)

    # 严谨求索者
    if df['official_replies'].sum() > 0:
        add_highlight("RIGOROUS_INQUIRER", "于 整个学期")

    # 互助典范
    peer_assists_df = df[df['peer_assists'] > 0]
    if not peer_assists_df.empty:
        # 获取第一个互助帖的标题用于展示
        first_assist_title = peer_assists_df.iloc[0]['assisted_post_titles'][0]
        add_highlight("PEER_MENTOR", f"于《{first_assist_title}》", post_title=first_assist_title)

    # [V8.7] 坚实奠基者
    if not strong_scores.empty and strong_scores.mean() < 75 and len(df) > 10:
        worst_hw = df.loc[strong_scores.idxmin()]
        add_highlight("FOUNDATION_BUILDER", f"于 {worst_hw['name']}", hw_name=worst_hw['name'])

    # [V8.7] 勤奋的探索者
    if not submit_times_df.empty:
        total_submissions = int(submit_times_df['public_test_used_times'].sum())
        if total_submissions > 30:
            add_highlight("DILIGENT_EXPLORER", "于 整个学期", total_submissions=total_submissions)

    # [V8.7] 坚韧不拔
    if len(strong_scores) > 1:
        for i in range(len(df) - 1):
            hw1, hw2 = df.iloc[i], df.iloc[i+1]
            s1, s2 = hw1.get('strong_test_score'), hw2.get('strong_test_score')
            if pd.notna(s1) and pd.notna(s2) and s1 < 80 and s2 - s1 > 15:
                add_highlight("THE_PERSEVERER", f"于 {hw2['name']}", low_score_hw=hw1['name'], rebound_hw=hw2['name'])
                break # 只记录第一次重大反弹

    # [V8.7] 积极的协作者
    if not mutual_df.empty:
        active_row = mutual_df.loc[mutual_df['hack_total_attempts'].idxmax(skipna=True)] if 'hack_total_attempts' in mutual_df.columns and not mutual_df['hack_total_attempts'].empty else None
        if active_row is not None and active_row['hack_total_attempts'] > 10:
             add_highlight("ACTIVE_COLLABORATOR", f"于 {active_row['name']}", hw_name=active_row['name'], hack_attempts=int(active_row['hack_total_attempts']))

    # [V8.0] 漏洞修复专家
    bugfix_df = df.dropna(subset=['bug_fix_hacked_count'])
    if not bugfix_df.empty and bugfix_df['bug_fix_hacked_count'].sum() > 0 and bugfix_df['bug_fix_unfixed_count'].sum() == 0:
        add_highlight("BUG_FIXER_PRO", "于 整个学期")

    # [V8.0] 并发挑战者
    unit2_df = df[df['unit'].str.contains("第二单元", na=False)]
    if not unit2_df.empty:
        has_perf_issues = any("TIME" in str(s) for s in unit2_df['strong_test_issues'].dropna())
        if has_perf_issues and unit2_df['strong_test_score'].mean() > 95:
             add_highlight("PERFORMANCE_CHALLENGER", "于 第二单元")

    # [V7.0] DDL逆袭者
    ddl_comeback_df = df[(df['start_ratio'] > 0.8) & (df['strong_test_score'] > 85)] # 使用 start_ratio
    if len(ddl_comeback_df) >= 2:
        add_highlight("DEADLINE_COMEBACK", f"于 {ddl_comeback_df.iloc[0]['name']}", hw_name=ddl_comeback_df.iloc[0]['name'])

    # 效率奇才
    if not submit_times_df.empty and not submit_times_df['public_test_used_times'].empty:
        min_submit_row = submit_times_df.loc[submit_times_df['public_test_used_times'].idxmin()]
        if min_submit_row['public_test_used_times'] <= 2:
            add_highlight("EFFICIENCY_ACE", f"于 {min_submit_row['name']}", hw_name=min_submit_row['name'])

    # 开局冲刺手
    early_submitters = df[df['start_ratio'] < 0.1]
    if len(early_submitters) >= 3:
        add_highlight("FAST_STARTER", f"于 {early_submitters.iloc[0]['name']}", hw_name=early_submitters.iloc[0]['name'])

    # 稳如磐石
    if not strong_scores.empty and not mutual_df.empty and strong_scores.min() > 95 and mutual_df['hacked_success'].sum() <= 1:
        add_highlight("ROCK_SOLID", "于 整个学期", min_score=strong_scores.min())

    # 防御大师
    if not mutual_df.empty and (mutual_df['hacked_success'] == 0).mean() >= 0.75:
        add_highlight("DEFENSE_MASTER", "于 整个学期")

    # 学霸本色
    if not strong_scores.empty and strong_scores.mean() > 98.5:
        add_highlight("TOP_SCORER", "于 整个学期", avg_score=strong_scores.mean())

    # 机会主义黑客
    if not mutual_df.empty and not mutual_df['hack_success'].empty:
        max_hack_row = mutual_df.loc[mutual_df['hack_success'].idxmax()]
        if max_hack_row['hack_success'] >= 10:
            add_highlight("HACK_ARTIST", f"于 {max_hack_row['name']}", hw_name=max_hack_row['name'], count=int(max_hack_row['hack_success']))

    style_df = df.dropna(subset=['style_score'])
    if not style_df.empty and (style_df['style_score'] == 100).all():
        add_highlight("CODE_ARTISAN", "于 整个学期")
        
    # --- 得分精算师 ---
    bugfix_df = df.dropna(subset=['bug_fix_hack_score', 'bug_fix_hacked_score'])
    if not bugfix_df.empty:
        total_hack_score = bugfix_df['bug_fix_hack_score'].sum()
        total_hacked_score = bugfix_df['bug_fix_hacked_score'].sum()
        # 避免除零，并确保有实际得分
        if total_hacked_score > 0 and total_hack_score > 0:
            ratio = total_hack_score / total_hacked_score
            if 0.9 <= ratio <= 1.1:
                add_highlight("SCORE_ACTUARY", "于 整个学期")
    # 编码马拉松选手
    if not df.empty and df['commit_count'].max() > 20:
        marathon_hw = df.loc[df['commit_count'].idxmax()]
        add_highlight("CODING_MARATHONER", f"于 {marathon_hw['name']}", 
                      hw_name=marathon_hw['name'], count=int(marathon_hw['commit_count']))
    
    # 主动重构者
    if df['commit_refactor_count'].sum() >= 3:
        add_highlight("PROACTIVE_REFACTORER", "于 整个学期", 
                      count=int(df['commit_refactor_count'].sum()))

    # 版本管理达人
    total_version_commits = df['commit_keywords'].apply(lambda x: x.get('versioning', 0)).sum()
    if total_version_commits >= 5:
        add_highlight("VERSION_CONTROL_GURU", "于 整个学期", count=int(total_version_commits))

    # 深夜码农
    night_commits = 0
    total_commits = 0
    for commits in df['commits']:
        if commits:
            total_commits += len(commits)
            for commit in commits:
                # 检查是否在凌晨1点到4点之间
                if 1 <= commit['timestamp'].hour <= 4:
                    night_commits += 1
    if total_commits > 0 and (night_commits / total_commits) > 0.3: # 如果超过30%的commit在深夜
        add_highlight("NIGHT_OWL_CODER", "于 整个学期", percentage=int((night_commits / total_commits) * 100))

    # 逐作业成就检查
    for _, hw in df.iterrows():
        # 众矢之的
        room_total_hacked = hw.get('room_total_hacked', 0)
        member_count = hw.get('room_member_count', 0)
        if room_total_hacked > 0 and member_count > 1 and (hw.get('hacked_success', 0) / room_total_hacked) > 0.4 and hw.get('hacked_total_attempts', 0) > (member_count / 2):
            add_highlight("PUBLIC_ENEMY_NO_1", f"于 {hw['name']}", hw_name=hw['name'])
        
        # --- 性能相关成就 (仅限第一、二单元) ---
        if hw['unit'].startswith("第一单元") or hw['unit'].startswith("第二单元"):
            details = hw.get('strong_test_details', [])
            if details: # 确保有详细数据
                scores = [r.get('score', 0) for r in details]
                if not scores: continue

                # 性能卓越者
                if all(s > 98 for s in scores):
                    add_highlight("PERFORMANCE_ACE", f"于 {hw['name']}", hw_name=hw['name'])
                
                # 正确性优先
                if all(85 <= s <= 92 for s in scores):
                    add_highlight("CORRECTNESS_FIRST", f"于 {hw['name']}", hw_name=hw['name'])

                # 性能赌徒
                if any(s == 0 for s in scores) and any(s > 98 for s in scores):
                    add_highlight("PERFORMANCE_GAMBLER", f"于 {hw['name']}", hw_name=hw['name'])
        # [V9.2 新增] 提取当前作业的关键信息
        my_events = hw.get('mutual_test_events', [])
        all_events = hw.get('room_events', [])
        end_time = hw.get('mutual_test_end_time')
        # [V9.3 新增] “团灭”成就判定
        member_count = hw.get('room_member_count', 0)
        target_count = hw.get('successful_hack_targets', 0)

        # 确保房间内不只有自己一个人
        if member_count > 1 and target_count == (member_count - 1):
            add_highlight("ANNIHILATION", f"于 {hw['name']}", hw_name=hw['name'])

        # --- 浴火重生 ---
        room_hacks = hw.get('room_total_hacked', 0)
        my_hacked = hw.get('hacked_success', 0)
        score = hw.get('strong_test_score', 0)
        if room_hacks > 25 and my_hacked > 0 and score > 90:
            add_highlight("PHOENIX_REBIRTH", f"于 {hw['name']}", hw_name=hw['name'])
            
        # --- 破冰者 (优化版) ---
        room_hack_success = hw.get('room_total_hack_success', 0)
        my_hack_success = hw.get('hack_success', 0)
        # 确保是“和平”局（总成功hack不多），且我方贡献了绝大部分
        if 0 < room_hack_success <= 5:
            if my_hack_success > 0 and (my_hack_success / room_hack_success) >= 0.8:
                add_highlight("ICE_BREAKER", f"于 {hw['name']}", hw_name=hw['name'])

        # --- 压哨绝杀 ---
        if my_events and pd.notna(end_time):
            for event in my_events:
                hack_time_str = event.get('submitted_at')
                if not hack_time_str: continue
                hack_time = pd.to_datetime(hack_time_str)
                # 检查时间差是否小于1小时 (3600秒)
                if pd.notna(hack_time) and (end_time - hack_time).total_seconds() < 3600:
                    add_highlight("BUZZER_BEATER", f"于 {hw['name']}", hw_name=hw['name'])
                    break # 每个作业只记录一次

        # --- 连锁反应 ---
        if my_events:
            # 按提交时间戳对成功的hack进行分组计数
            submission_counts = Counter(e.get('submitted_at') for e in my_events if e.get('submitted_at'))
            if submission_counts:
                top_submission = submission_counts.most_common(1)[0]
                if top_submission[1] >= 3: # 如果单次提交成功次数 >= 3
                    add_highlight("CHAIN_REACTION", f"于 {hw['name']}", hw_name=hw['name'], count=top_submission[1])

        # --- 反戈一击 ---
        if all_events:
            was_hacked = False
            hacked_time = pd.Timestamp.min.tz_localize('UTC') if pd.Timestamp.min.tz is None else pd.Timestamp.min
            
            # 确保 all_events 按时间排序
            sorted_events = sorted(all_events, key=lambda x: x.get('submitted_at', ''))
            
            for event in sorted_events:
                event_time_str = event.get('submitted_at')
                if not event_time_str: continue
                event_time = pd.to_datetime(event_time_str)

                # 如果我被攻击了，记录时间和状态
                if is_target_user(event.get('hacked', {}), config):
                    was_hacked = True
                    hacked_time = event_time
                
                # 如果我之前被攻击过，并且现在我攻击了别人
                if was_hacked and is_target_user(event.get('hack', {}), config):
                    # 确保这次攻击发生在被攻击之后
                    if event_time > hacked_time:
                        add_highlight("COUNTER_ATTACK", f"于 {hw['name']}", hw_name=hw['name'])
                        break # 找到一次反击即可

        # [V9.0 新增] 第一滴血成就判定
        all_events = hw.get('room_events', [])
        if all_events and is_target_user(all_events[0].get('hack', {}), config):
            add_highlight("FIRST_BLOOD", f"于 {hw['name']}", hw_name=hw['name'])

        # [新增] 规划大师
        is_planning_master = (hw.get('ddl_index', 1) < 0.3 and hw.get('public_test_used_times', 99) <= 2 and
                              hw.get('strong_test_score', 0) == 100 and hw.get('hacked_success', 99) == 0)
        if is_planning_master:
            add_highlight("PLANNING_MASTER", f"于 {hw['name']}", hw_name=hw['name'], count=int(hw['public_test_used_times']))

        # [新增] 铁壁小队成员
        is_iron_wall = (hw.get('room_total_hack_success', 99) == 0 and hw.get('room_total_hack_attempts', 0) > 50)
        if is_iron_wall:
            add_highlight("IRON_WALL_SQUAD", f"于 {hw['name']}", hw_name=hw['name'], total_attacks=int(hw['room_total_hack_attempts']))

        # [V8.9 修正逻辑] 精准打击者
        if pd.notna(hw.get('hack_success_rate')) and hw['hack_success_rate'] > 8:
            add_highlight("PRECISION_STRIKER", f"于 {hw['name']}", hw_name=hw['name'], rate=hw['hack_success_rate'])

        # [V9.0 优化] “战术大师”的判定，现在直接基于事件日志，更准确
        targets = hw.get('successful_hack_targets', 0)
        successes = hw.get('hack_success', 0)
        if targets > 0 and successes > 3 and (successes / targets) > 1.8: # 使用更严格的比率
            add_highlight("TACTICAL_MASTER", f"于 {hw['name']}", hw_name=hw['name'], target_count=int(targets), hack_count=int(successes))

        # [V8.5] 风暴幸存者
        if hw.get('room_total_hacked', 0) > 20 and hw.get('hacked_success', 100) <= 1:
            add_highlight("STORM_SURVIVOR", f"于 {hw['name']}", hw_name=hw['name'], room_total_hacked=int(hw['room_total_hacked']), self_hacked=int(hw['hacked_success']))

    # 表达式大师
    unit1_df = df[df['unit'].str.contains("第一单元", na=False)]
    if not unit1_df.empty and unit1_df['strong_test_score'].mean() > 98 and unit1_df['hacked_success'].sum() <= 4:
        add_highlight("EXPRESSION_GURU", "于 第一单元")

    # [新增] 并发指挥家 (第二单元)
    unit2_df = df[df['unit'].str.contains("第二单元", na=False)]
    if not unit2_df.empty and unit2_df['strong_test_score'].mean() > 95 and unit2_df['hacked_success'].sum() <= 8:
        # 复用之前计算过的unit2_df
        add_highlight("CONCURRENCY_CONDUCTOR", "于 第二单元")

    # JML大师
    unit3_df = df[df['unit'].str.contains("第三单元", na=False)]
    if not unit3_df.empty and unit3_df['strong_test_score'].mean() > 99 and unit3_df['hacked_success'].sum() == 0:
        add_highlight("JML_MASTER", "于 第三单元")

    # UML专家
    unit4_df = df[df['unit'].str.contains("第四单元", na=False)]
    if not unit4_df.empty and unit4_df['strong_test_score'].mean() == 100:
        is_perfect = all(all(r['message'] == 'ACCEPTED' for r in row.get('uml_detailed_results', [])) for _, row in unit4_df.iterrows() if row.get('uml_detailed_results'))
        if is_perfect:
            add_highlight("UML_EXPERT", "于 第四单元")

    # [V8.0] 架构迭代大师
    for unit_name in df['unit'].unique():
        unit_df = df[df['unit'] == unit_name].sort_values('hw_num')
        if len(unit_df) > 1:
            scores = unit_df['strong_test_score'].dropna()
            if len(scores) > 1 and scores.iloc[-1] - scores.iloc[0] > 10 and scores.iloc[-1] > 95:
                unit_name_short = re.sub(r'：.*', '', unit_name)
                add_highlight("REFACTOR_VIRTUOSO", f"于 {unit_name_short}", unit_name=unit_name_short, hw_name_before=unit_df.iloc[0]['name'], hw_name_after=unit_df.iloc[-1]['name'])
                break # 每个单元只记录一次

    # [V8.0] 王者归来
    unit_scores = df.groupby('unit')['strong_test_score'].mean()
    u1_key, u4_key = "第一单元：表达式求导", "第四单元：UML解析"
    if u1_key in unit_scores and u4_key in unit_scores:
        u1_score, u4_score = unit_scores[u1_key], unit_scores[u4_key]
        if pd.notna(u1_score) and pd.notna(u4_score) and u4_score > u1_score + 2:
            add_highlight("COMEBACK_KING", "于 整个学期", u1_score=u1_score, u4_score=u4_score)

    # --- 2. 元成就计算 ---
    if len(earned_achievements) > 5:
        add_highlight("DECORATED_DEVELOPER", "于 整个学期")

    all_possible_keys = set(REPORT_CORPUS["HIGHLIGHTS"]["TAGS"].keys())
    keys_to_check_for_mastery = all_possible_keys - {"COLLECTION"}
    if len(earned_achievements) >= len(keys_to_check_for_mastery) * 0.7:
        add_highlight("COLLECTION", "于 整个学期")

    # --- 3. 随机选择5个亮点用于报告主体展示 ---
    if not earned_achievements:
        return [], {}

    highlight_texts_to_choose_from = [d['description'] for d in earned_achievements.values()]
    random.shuffle(highlight_texts_to_choose_from)
    final_highlights_for_display = highlight_texts_to_choose_from[:5]

    # --- 4. 返回最终结果 ---
    return final_highlights_for_display, earned_achievements

def identify_student_persona(df):
    """[V8.7 改造] 优先为成绩不理想的同学选择更温和的画像"""
    if df.empty: return "BALANCED"
    
    strong_scores = df['strong_test_score'].dropna()
    avg_score = strong_scores.mean() if not strong_scores.empty else 100
    
    # [V8.7 新增] 如果平均分较低，优先使用鼓励性Persona
    if avg_score < 75:
        return "BALANCED_GENTLE"

    mutual_df = df[df.get('has_mutual_test', pd.Series(False))]
    if 'start_ratio' in df.columns and df['start_ratio'].dropna().mean() > 0.7: return "SPRINTER"
    if not mutual_df.empty and mutual_df['hack_success'].sum() > 25: return "HUNTER"
    if (not mutual_df.empty and mutual_df['hacked_success'].sum() <= 3) and df['strong_test_score'].var() < 10: return "FORTRESS"
    if df['public_test_used_times'].dropna().mean() > 6: return "GRINDER"
    return "BALANCED"

def format_uml_analysis(hw_row):
    uml_results = hw_row.get('uml_detailed_results', [])
    if not uml_results: return ""
    failed_checks = [r['name'] for r in uml_results if r['message'] != 'ACCEPTED']
    if not failed_checks: return random.choice(REPORT_CORPUS["UML"]["PERFECT"])
    else: return random.choice(REPORT_CORPUS["UML"]["IMPERFECT"]).format(issues=', '.join(failed_checks))

def _analyze_overall_performance(df):
    """[V8.7-Patched] 辅助函数，生成宏观表现的文字分析，加入对C房和低分情况的同理心分析"""
    analysis_texts = []
    
    strong_scores = df['strong_test_score'].dropna()
    avg_score = strong_scores.mean() if not strong_scores.empty else 0

    if not strong_scores.empty:
        var_score = strong_scores.var()
        analysis_texts.append(f"强测表现: 平均分 {avg_score:.2f} | 稳定性 (方差) {var_score:.2f}")
        
        # --- [V8.7 改造] 强测表现分析优化 ---
        if avg_score > 98:
            analysis_texts.append(random.choice(REPORT_CORPUS["ANALYSIS"]["STRONG_TEST"]["HIGH_SCORE"]))
        elif avg_score < 75:
            struggle_hws = df[df['strong_test_score'] < 70]['name'].tolist()
            if struggle_hws:
                 analysis_texts.append(random.choice(REPORT_CORPUS["ANALYSIS"]["STRONG_TEST"]["STRUGGLE"]).format(hw_names=', '.join(struggle_hws)))
        else:
            imperfect_hws = df[df['strong_test_score'] < 100]['name'].tolist()
            if imperfect_hws:
                analysis_texts.append(random.choice(REPORT_CORPUS["ANALYSIS"]["STRONG_TEST"]["IMPERFECTION"]).format(hw_names=', '.join(imperfect_hws[:2])))
        
        if var_score < 15:
            if avg_score > 80: # 1. 高分稳定 -> 赞美
                analysis_texts.append(random.choice(REPORT_CORPUS["ANALYSIS"]["CONSISTENCY"]["STABLE"]).format(variance=var_score))
            else: # 2. 低分稳定 -> 鼓励寻求突破 (调用新增语料库)
                analysis_texts.append(random.choice(REPORT_CORPUS["ANALYSIS"]["CONSISTENCY"]["LOW_SCORE_STABLE"]).format(variance=var_score))
        else:
            best_hw = df.loc[df['strong_test_score'].idxmax()]['name'] if pd.notna(df['strong_test_score'].max()) else '某次作业'
            worst_hw = df.loc[df['strong_test_score'].idxmin()]['name'] if pd.notna(df['strong_test_score'].min()) else '另一次作业'
            analysis_texts.append(random.choice(REPORT_CORPUS["ANALYSIS"]["CONSISTENCY"]["VOLATILE"]).format(variance=var_score, best_hw=best_hw, worst_hw=worst_hw))

    mutual_df = df[df.get('has_mutual_test', pd.Series(False))].dropna(subset=['hack_success', 'hacked_success', 'hacked_total_attempts'])
    if not mutual_df.empty:
        total_hacks = mutual_df['hack_success'].sum()
        total_hacked = mutual_df['hacked_success'].sum()
        total_hacked_attempts = mutual_df['hacked_total_attempts'].sum()
        total_hacks_attempted = mutual_df['hack_total_attempts'].sum()
        
        analysis_texts.append(f"\n互测战绩: 发起攻击 {int(total_hacks_attempted)} 次 (成功 {int(total_hacks)}) | 被攻击 {int(total_hacked_attempts)} 次 (被成功Hack {int(total_hacked)})")

        if total_hacks_attempted > 0:
            rate = (total_hacks / total_hacks_attempted) * 100
            # 设置高攻击次数的阈值，例如100次
            if total_hacks_attempted > 100:
                # 设置高效率的阈值，例如8%
                if rate > 8:
                    analysis_texts.append(random.choice(REPORT_CORPUS["MUTUAL_TEST"]["ATTACK_STYLE"]["PROLIFIC_ATTACKER"]).format(
                        attempts=int(total_hacks_attempted), rate=rate
                    ))
                else:
                    analysis_texts.append(random.choice(REPORT_CORPUS["MUTUAL_TEST"]["ATTACK_STYLE"]["PERSISTENT_ATTACKER"]).format(
                        attempts=int(total_hacks_attempted), successes=int(total_hacks)
                    ))
            else: # 攻击次数不多
                analysis_texts.append(random.choice(REPORT_CORPUS["MUTUAL_TEST"]["ATTACK_STYLE"]["SELECTIVE_ATTACKER"]).format(
                    attempts=int(total_hacks_attempted)
                ))

        profile_found = False
        if total_hacks > total_hacked * 2 and total_hacks >= 10:
            hw_with_most_hacks = mutual_df.loc[mutual_df['hack_success'].idxmax()]
            total_unique_targets = mutual_df['successful_hack_targets'].sum()
            total_hack_attempts = mutual_df['hack_total_attempts'].sum()
            overall_hack_rate = (total_hacks / total_hack_attempts) * 100 if total_hack_attempts > 0 else 0
            offensive_format_vars = {
                'hw_name_most_hacks': hw_with_most_hacks['name'], 'hacks_in_best_hw': int(hw_with_most_hacks['hack_success']),
                'total_hacks': int(total_hacks), 'total_unique_targets': int(total_unique_targets), 'overall_hack_rate': overall_hack_rate
            }
            analysis_texts.append(random.choice(REPORT_CORPUS["MUTUAL_TEST"]["OFFENSIVE"]).format(**offensive_format_vars))
            profile_found = True
        
        if total_hacked <= 3:
            analysis_texts.append(random.choice(REPORT_CORPUS["MUTUAL_TEST"]["DEFENSIVE"]).format(count=int(total_hacked_attempts), hw_names="各次", hacked="多"))
            profile_found = True
        
        if total_hacked_attempts > 20: 
            hacked_rate = (total_hacked / total_hacked_attempts) * 100 if total_hacked_attempts > 0 else 0
            if hacked_rate < 15:
                analysis_texts.append(random.choice(REPORT_CORPUS["MUTUAL_TEST"]["BATTLE_HARDENED"]).format(
                    total_hacked_attempts=int(total_hacked_attempts), total_hacked=int(total_hacked), rate=hacked_rate))
                profile_found = True
        
        if not profile_found:
            analysis_texts.append(random.choice(REPORT_CORPUS["MUTUAL_TEST"]["BALANCED"]).format(total_hacks=int(total_hacks), total_hacked=int(total_hacked)))

        # --- [V8.7 改造] 互测相对表现分析优化 ---
        room_df = df.dropna(subset=['room_level'])
        if not room_df.empty:
            a_room_rate = (room_df['room_level'] == 'A').mean()
            c_room_rate = (room_df['room_level'] == 'C').mean()
            if a_room_rate > 0.6:
                analysis_texts.append(random.choice(REPORT_CORPUS["MUTUAL_TEST"]["RELATIVE_PERFORMANCE"]["A_ROOM"]).format(a_room_rate=a_room_rate*100))
            elif c_room_rate > 0.6:
                analysis_texts.append(random.choice(REPORT_CORPUS["MUTUAL_TEST"]["RELATIVE_PERFORMANCE"]["C_ROOM"]))
            else:
                analysis_texts.append(random.choice(REPORT_CORPUS["MUTUAL_TEST"]["RELATIVE_PERFORMANCE"]["MIXED"]))

        total_weighted_deduction = mutual_df['weighted_defense_deduction'].sum()
        max_possible_deduction = mutual_df.shape[0] * 10 
        defense_score = 100 - (total_weighted_deduction / max(max_possible_deduction, 1) * 10) # 修正了计算公式
        analysis_texts.append(random.choice(REPORT_CORPUS["ANALYSIS"]["DEFENSE_SCORE"]).format(score=max(0, defense_score)))

    unit2_df = df[df['unit'].str.contains("第二单元", na=False)]
    if not unit2_df.empty:
        perf_issues = {}
        for _, row in unit2_df.iterrows():
            for issue, count in row.get('strong_test_issues', {}).items():
                if "TIME_LIMIT_EXCEED" in issue:
                    perf_issues.setdefault(issue, []).append(row['name'])
        if perf_issues:
            issue_types = ", ".join(perf_issues.keys())
            hw_names = ", ".join(list(set(sum(perf_issues.values(), []))))
            analysis_texts.append("\n" + random.choice(REPORT_CORPUS["ANALYSIS"]["PERFORMANCE_ISSUE"]).format(hw_names=hw_names, issue_types=issue_types))
            
    analysis_texts.append("\n" + "开发习惯洞察:") # 添加一个清晰的子标题
    commit_df = df[df['commit_count'] > 2].copy()
    
    if commit_df.empty:
        analysis_texts.append(random.choice(REPORT_CORPUS["OVERALL_DEV_PROCESS"]["NO_DATA"]))
    else:
        # 1. 总体介绍
        avg_commits = commit_df['commit_count'].mean()
        total_refactors = int(commit_df['commit_refactor_count'].sum())
        
        if total_refactors > 0:
            # 如果有重构记录，使用 INTRO 语料
            intro_corpus = REPORT_CORPUS["OVERALL_DEV_PROCESS"]["INTRO"]
            analysis_texts.append(random.choice(intro_corpus).format(
                avg_commits=avg_commits,
                refactor_count=total_refactors
            ))
        else:
            # 如果没有重构记录，使用 INTRO_NO_REFACTOR 语料
            intro_corpus = REPORT_CORPUS["OVERALL_DEV_PROCESS"]["INTRO_NO_REFACTOR"]
            analysis_texts.append(random.choice(intro_corpus).format(
                avg_commits=avg_commits
            ))
    
        # 2. 总体启动风格
        avg_start_ratio = commit_df['start_ratio'].mean()
        style_key = ""
        if avg_start_ratio < 0.3:
            style_key = "EARLY_BIRD"
        elif avg_start_ratio > 0.6:
            style_key = "DDL_FIGHTER"
        else:
            style_key = "WELL_PACED"

        # 从新的语料库中选择风格描述
        if style_key:
            style_corpus = REPORT_CORPUS["OVERALL_DEV_PROCESS"]["STYLE"][style_key]
            analysis_texts.append(random.choice(style_corpus))

    return analysis_texts

def _analyze_hack_strategy(df):
    """[V9.0 升级] 分析互测博弈策略，包含时机、目标选择和有效性"""
    texts = []
    # 筛选出包含有效互测事件和相关时间的作业
    mutual_df = df.dropna(subset=['mutual_test_start_time', 'mutual_test_end_time', 'hack_success_rate'])
    # 确保 mutual_test_events 列非空
    mutual_df = mutual_df[mutual_df['mutual_test_events'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    
    if mutual_df.empty:
        return []

    # --- 1. 时机分析 (Timing Analysis) ---
    early_hacks, late_hacks, total_hack_events = 0, 0, 0
    for _, hw in mutual_df.iterrows():
        duration = (hw['mutual_test_end_time'] - hw['mutual_test_start_time']).total_seconds()
        if duration <= 0: continue
        
        events = hw.get('mutual_test_events', [])
        total_hack_events += len(events)
        
        for event in events:
            # 确保 event 是字典并且包含 'time' 键
            # [修正] 旧版数据格式为 {'time':...}, 新版为 {'submitted_at':...}
            # 为了兼容，我们直接从 event 中获取时间
            event_time_str = event.get('submitted_at')
            if not event_time_str: continue

            hack_time = pd.to_datetime(event_time_str)
            ratio = (hack_time - hw['mutual_test_start_time']).total_seconds() / duration
            if ratio < 0.1: early_hacks += 1
            if ratio > 0.9: late_hacks += 1

    if total_hack_events > 0:
        if early_hacks / total_hack_events > 0.5:
            texts.append(REPORT_CORPUS["HACK_STRATEGY"]["TIMING"]["EARLY_BIRD"])
        elif late_hacks / total_hack_events > 0.5:
            texts.append(REPORT_CORPUS["HACK_STRATEGY"]["TIMING"]["DEADLINE_SNIPER"])
        else:
            texts.append(REPORT_CORPUS["HACK_STRATEGY"]["TIMING"]["CONSISTENT_PRESSURE"])

    # --- 2. 目标选择分析 (Targeting Analysis) ---
    texts.append("") # 添加一个空行用于分隔
    focused_fire_count = 0
    wide_net_count = 0
    for _, hw in mutual_df.iterrows():
        my_events = hw.get('mutual_test_events', [])
        if not my_events: continue
        
        total_hacks = len(my_events)
        
        # [V9.0 修正] 从完整的事件结构中正确提取目标ID
        # 这里是关键的修复点：不再使用 e['target']
        unique_targets = len(set(e['hacked']['student_id'] for e in my_events if 'hacked' in e and 'student_id' in e['hacked']))
        
        if total_hacks > 2 and unique_targets > 0:
            if (total_hacks / unique_targets) > 1.8:
                focused_fire_count += 1
            else:
                wide_net_count += 1
    
    if focused_fire_count > wide_net_count:
        texts.append(REPORT_CORPUS["HACK_STRATEGY"]["TARGETING"]["FOCUSED_FIRE"])
    elif wide_net_count > 0:
        texts.append(REPORT_CORPUS["HACK_STRATEGY"]["TARGETING"]["WIDE_NET"])
        
    # --- 3. 有效性分析 (Effectiveness Analysis) ---
    texts.append("")
    total_hacks_ever = mutual_df['hack_success'].sum()
    avg_effectiveness_rate = mutual_df['hack_success_rate'].mean()
    if pd.notna(avg_effectiveness_rate):
        if avg_effectiveness_rate > 8:
            texts.append(REPORT_CORPUS["HACK_STRATEGY"]["EFFECTIVENESS"]["HIGH_EFFICIENCY"])
        elif total_hacks_ever > 0:
            texts.append(REPORT_CORPUS["HACK_STRATEGY"]["EFFECTIVENESS"]["PERSISTENT_EFFORT"])

    return texts

# 在 analyze.py 中

def format_development_process_for_hw(hw_row):
    """[V3.0 重构] 为单次作业格式化开发流程的描述文本，智能复用现有语料库"""
    tags = hw_row.get('dev_style_tags', [])
    if not tags:
        return ""
    main_style_tag = tags[0]  # 第一个标签作为主要风格
    additional_tags = tags[1:]
    # 1. 获取主要工作模式的描述
    # 复用 SUBMISSION.STYLE 语料库
    main_desc = ""
    if main_style_tag in REPORT_CORPUS["SUBMISSION"]["STYLE"]:
        main_desc = random.choice(REPORT_CORPUS["SUBMISSION"]["STYLE"][main_style_tag])
    
    if not main_desc:
        return "" # 如果没有主要描述，则不生成

    # 2. 获取附加过程的描述
    additional_descs = []
    for tag in additional_tags:
        # 使用新的 HW_DEV_PROCESS 语料库获取补充描述
        if tag in REPORT_CORPUS.get("HW_DEV_PROCESS", {}):
            additional_descs.append(random.choice(REPORT_CORPUS["HW_DEV_PROCESS"][tag]))

    # 3. 智能组合描述
    # 使用连接词语料
    connectors = REPORT_CORPUS["HW_DEV_PROCESS"]["CONNECTORS"]
    
    if additional_descs:
        # e.g., "本次作业你的开发风格是：[主要风格]。此外，你还对代码进行了重构，追求卓越。"
        return random.choice(connectors["WITH_ADDITIONS"]).format(
            main_desc=main_desc,
            additions="，".join(additional_descs)
        )
    else:
        # e.g., "本次作业你的开发风格是：[主要风格]。"
        return random.choice(connectors["SINGLE_MAIN"]).format(main_desc=main_desc)

def generate_dynamic_report(df, user_name, config):
    print("\n" + "="*80)
    print(f" {user_name} - OO课程动态学习轨迹报告 V8.9 ".center(80, "="))
    print("="*80)

    if df.empty:
        print("\n未找到该学生的有效作业数据，请检查配置文件。")
        return

    persona_key = identify_student_persona(df)
    persona_text = REPORT_CORPUS["PERSONA"].get(persona_key, REPORT_CORPUS["PERSONA"]["BALANCED"])
    print("\n" + persona_text.format(user_name=user_name))

    # [修改] 接收新的返回值
    highlights_for_display, earned_achievements_details = generate_highlights(df, config)
    earned_highlight_keys = set(earned_achievements_details.keys())

    if highlights_for_display:
        print("\n" + "--- 1. 个人亮点标签 ---".center(70))
        print(random.choice(REPORT_CORPUS["HIGHLIGHTS"]["INTRO"]))
        for tag_text in highlights_for_display:
            print(tag_text)

    # --- 报告主体部分 (2-7) ---
    print("\n" + "--- 2. 宏观学期表现与深度洞察 ---".center(70))
    for text in _analyze_overall_performance(df):
        print(text)

    print("\n" + "--- 3. 开发者责任感与调试能力 (Bug修复) ---".center(70))
    bugfix_df = df.dropna(subset=['bug_fix_hacked_count'])
    total_bugs = bugfix_df['bug_fix_hacked_count'].sum()
    if total_bugs > 0:
        fixed_bugs = total_bugs - bugfix_df['bug_fix_unfixed_count'].sum()
        fix_rate = (fixed_bugs / total_bugs) * 100
        print(REPORT_CORPUS["BUG_FIX"]["ANALYSIS"]["HIGH_FIX_RATE" if fix_rate > 80 else "LOW_FIX_RATE"].format(total_bugs=int(total_bugs), fixed_bugs=int(fixed_bugs), rate=fix_rate))

        total_hack_score, total_hacked_score = bugfix_df['bug_fix_hack_score'].sum(), bugfix_df['bug_fix_hacked_score'].sum()
        if total_hack_score + total_hacked_score > 0:
            ratio = (total_hack_score + 0.1) / (total_hacked_score + 0.1)
            if ratio > 1.5: print(REPORT_CORPUS["BUG_FIX"]["ANALYSIS"]["HACK_FOCUSED"].format(hack_score=total_hack_score, hacked_score=total_hacked_score, ratio=ratio))
            elif ratio < 0.7: print(REPORT_CORPUS["BUG_FIX"]["ANALYSIS"]["FIX_FOCUSED"].format(hack_score=total_hack_score, hacked_score=total_hacked_score, ratio=ratio))
        print(random.choice(REPORT_CORPUS["BUG_FIX"]["INSIGHT"]))
    else:
        print(REPORT_CORPUS["BUG_FIX"]["ANALYSIS"]["NO_BUGS_TO_FIX"])

    print("\n" + "--- 4. 单元深度与成长轨迹 ---".center(70))
    unit_paradigms = {"第一单元": "递归下降", "第二单元": "多线程", "第三单元": "JML规格", "第四单元": "UML解析"}
    for unit_name_full, hw_nums in config["UNIT_MAP"].items():
        unit_df = df[df['unit'] == unit_name_full]
        unit_name_short = re.sub(r'：.*', '', unit_name_full)
        if not unit_df.empty and pd.notna(unit_df['strong_test_score'].mean()):
            print(random.choice(REPORT_CORPUS["ANALYSIS"]["UNIT"]).format(unit_name=unit_name_short, avg_score=unit_df['strong_test_score'].mean(), unit_paradigm=unit_paradigms.get(unit_name_short, "核心技术"), hacks=int(unit_df['hack_success'].sum()), hacked=int(unit_df['hacked_success'].sum())))

    strong_scores = df['strong_test_score'].dropna()
    if len(strong_scores) > 8:
        early_avg, later_avg = strong_scores.iloc[:len(strong_scores)//2].mean(), strong_scores.iloc[len(strong_scores)//2:].mean()
        if later_avg > early_avg + 1:
            print(random.choice(REPORT_CORPUS["ANALYSIS"]["GROWTH"]).format(early_avg=early_avg, later_avg=later_avg))

    print("\n" + "--- 5. 提交行为与风险分析 ---".center(70))
    print(random.choice(REPORT_CORPUS["SUBMISSION"]["INTRO"]))
    submit_times_df = df.dropna(subset=['public_test_used_times'])
    if not submit_times_df.empty and not submit_times_df['public_test_used_times'].empty:
        total_submissions = submit_times_df['public_test_used_times'].sum()
        print(f"本学期你共提交 {int(total_submissions)} 次代码。")
        most_submitted, least_submitted = submit_times_df.loc[submit_times_df['public_test_used_times'].idxmax()], submit_times_df.loc[submit_times_df['public_test_used_times'].idxmin()]
        print(random.choice(REPORT_CORPUS["SUBMISSION"]["MOST"]).format(hw_name=most_submitted['name'], count=int(most_submitted['public_test_used_times'])))
        print(random.choice(REPORT_CORPUS["SUBMISSION"]["LEAST"]).format(hw_name=least_submitted['name'], count=int(least_submitted['public_test_used_times'])))

    ddl_risk_df = df[(df['ddl_index'] > 0.9) & ((df['strong_test_deduction_count'] > 0) | (df['hacked_success'] > 0))]
    if len(ddl_risk_df) > len(df) * 0.1:
        print(random.choice(REPORT_CORPUS["ANALYSIS"]["DDL"]))

    hack_strategy_texts = _analyze_hack_strategy(df)
    if hack_strategy_texts:
        print("\n" + "--- 6. 互测博弈策略分析 ---".center(70))
        print(random.choice(REPORT_CORPUS["HACK_STRATEGY"]["INTRO"]))
        for text in hack_strategy_texts:
            print(text)

    total_essentials = df['essential_posts_authored'].sum()
    total_official_replies = df['official_replies'].sum()
    total_peer_assists = df['peer_assists'].sum()

    # 只有当学生有任何一种高质量互动时，才打印本章节
    if total_essentials > 0 or total_official_replies > 0 or total_peer_assists > 0:
        print("\n" + "--- [特别洞察] 社区互动与学习风格 ---".center(70))
        print(random.choice(REPORT_CORPUS["FORUM_ANALYSIS"]["INTRO"]))

        # 分析并打印“社区之光”行为
        if total_essentials > 0:
            # 安全地获取第一个精华帖的标题用于展示
            first_essential_title = df[df['essential_posts_authored'] > 0].iloc[0]['essential_post_titles'][0]
            print(random.choice(REPORT_CORPUS["FORUM_ANALYSIS"]["COMMUNITY_PILLAR_TEXT"]).format(
                post_title=first_essential_title
            ))
        
        # 分析并打印“严谨求索者”行为
        if total_official_replies > 0:
            print(random.choice(REPORT_CORPUS["FORUM_ANALYSIS"]["RIGOROUS_INQUIRER_TEXT"]).format(
                count=int(total_official_replies)
            ))
        
        # 分析并打印“互助典范”行为
        if total_peer_assists > 0:
            print(random.choice(REPORT_CORPUS["FORUM_ANALYSIS"]["PEER_MENTOR_TEXT"]).format(
                count=int(total_peer_assists)
            ))
    else:
    #     # 如果上面没有打印任何内容，可以考虑在这里打印"NO_ACTIVITY"
        print("\n" + "--- [特别洞察] 社区互动与学习风格 ---".center(70))
        print(random.choice(REPORT_CORPUS["FORUM_ANALYSIS"]["NO_ACTIVITY"]))
    peace_room_text_generated = False
    for _, hw in df.iterrows():
        if hw.get('room_total_hack_success', 99) == 0 and hw.get('room_total_hack_attempts', 0) > 50:
            if not peace_room_text_generated:
                print("\n" + "--- [特别洞察] 房间生态分析 ---".center(70))
                peace_room_text_generated = True
            print(random.choice(REPORT_CORPUS["MUTUAL_TEST"]["ROOM_ECOLOGY"]["PEACE_ROOM"]).format(
                hw_name=hw['name'],
                total_attacks=int(hw['room_total_hack_attempts'])
            ))

    print("\n" + "--- 7. 逐次作业深度解析 ---".center(70))
    print(random.choice(REPORT_CORPUS["ANALYSIS"]["HW_INTRO"]))
    for _, hw in df.iterrows():
        print(f"\n--- {hw['name']} ---")
        if pd.notna(hw.get('strong_test_score')):
            score_str = f"  - 强测: {hw.get('strong_test_score'):.2f}"
            if hw.get('strong_test_deduction_count', 0) > 0:
                issue_str = ", ".join([f"{k}({v}次)" for k,v in hw.get('strong_test_issues', {}).items()])
                score_str += f" | 扣分: {issue_str}"
            print(score_str)
        if hw.get('has_mutual_test') and pd.notna(hw.get('hack_success')):
            hack_info = f"Hack {int(hw.get('hack_success', 0))} | 被成功Hack {int(hw.get('hacked_success', 0))} (被攻击 {int(hw.get('hacked_total_attempts', 0))} 次)"
            if pd.notna(hw.get('room_avg_hacked')):
                 hack_info += f" (房均被Hack: {hw.get('room_avg_hacked', 0):.2f})"
            print(f"  - 互测: 在 {hw.get('room_level', '?')} 房化身「{hw.get('alias_name', '?')}」，{hack_info}")

        if hw['unit'].startswith("第四单元"):
            print(format_uml_analysis(hw))
        print(f"  - {format_development_process_for_hw(hw)}")
        if pd.notna(hw['ddl_index']):
            ddl_index = hw['ddl_index']
            if ddl_index > 0.9:
                delivery_key = 'FINAL_MOMENT'
            elif ddl_index < 0.3:
                delivery_key = 'EARLY_STAGE'
            else:
                delivery_key = 'MID_STAGE'
            
            # 从语料库中随机选择一条描述
            delivery_desc = random.choice(REPORT_CORPUS["HW_DELIVERY_STYLE"][delivery_key])
            print(f"  - 最终交付: {delivery_desc}")

    # --- [新功能] 成就墙 (V2.0 新版格式) ---
    print("\n" + "--- 8. 个人成就墙 ---".center(70))

    all_achievements_data = REPORT_CORPUS["HIGHLIGHTS"]["TAGS"]
    total_achievements = len(all_achievements_data)
    completed_achievements = len(earned_highlight_keys)
    
    print(f"成就进度：{completed_achievements} / {total_achievements}")

    unlocked_list = []
    locked_list = []

    for key, data in all_achievements_data.items():
        if key in earned_highlight_keys:
            unlocked_list.append((key, data))
        else:
            locked_list.append((key, data))

    # 按名称排序
    unlocked_list.sort(key=lambda x: x[1]['name'])
    locked_list.sort(key=lambda x: x[1]['name'])
    
    # 打印已解锁成就
    if unlocked_list:
        print("\n--- ✅ 已解锁成就 ---")
        for key, data in unlocked_list:
            icon = data.get('icon', '❓')
            name = data.get('name', '未知成就')
            context_info = earned_achievements_details[key].get('context', '')
            print(f"  {icon} {name} (达成于: {context_info.strip()})")

    # 打印未解锁成就
    if locked_list:
        print("\n--- 🔒 未解锁成就 ---")
        for key, data in locked_list:
            icon = data.get('icon', '❓')
            name = data.get('name', '未知成就')
            condition = data.get('condition', '未知条件')
            print(f"  {icon} {name} - {condition}")


    print("\n" + "="*80)
    print(" 学期旅程总结 ".center(80, "="))
    print("="*80)
    print(random.choice(REPORT_CORPUS["CONCLUSION"]))


# --- 7. 主执行逻辑 ---
def main():
    try:
        try:
            with open(CONFIG['YAML_CONFIG_PATH'], 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            student_id = str(yaml_config.get('stu_id'))
            if not student_id or not student_id.isdigit():
                raise ValueError("学号无效或缺失")
        except FileNotFoundError:
            exit_with_error(f"配置文件 '{CONFIG['YAML_CONFIG_PATH']}' 未找到。请确保它在当前目录。")
        except (yaml.YAMLError, ValueError) as e:
            exit_with_error(f"配置文件 '{CONFIG['YAML_CONFIG_PATH']}' 格式错误或内容无效: {e}")

        # read in DataFrame
        try:
            df = pd.read_pickle(CONFIG["FILE_PATH"])
        except pd.errors.ParserError as e:
            exit_with_error(f"数据文件 '{CONFIG["FILE_PATH"]}' 遇到错误: {e}")
        
        try:
            CONFIG["USER_INFO"].update(json.load(
                open(CONFIG["USER_PATH"], "r", encoding="utf-8")
            ))
        except FileNotFoundError:
            exit_with_error(f"配置文件 '{CONFIG["USER_PATH"]}' 未找到。请确保它在当前目录。")
        except (yaml.YAMLError, ValueError) as e:
            exit_with_error(f"配置文件 '{CONFIG["USER_PATH"]}' 格式错误或内容无效: {e}")

        try:
            curpos_path = Path("tools", "corpus.json")
            global REPORT_CORPUS
            REPORT_CORPUS = json.load(open(curpos_path, "r", encoding="utf-8"))
        except FileNotFoundError:
            exit_with_error(f"语料库文件 '{curpos_path}' 未找到。请确保它位于 'tools' 子目录中。")
        except json.JSONDecodeError:
            exit_with_error(f"语料库文件 '{curpos_path}' 格式错误。")
        
        df = enrich(df, CONFIG)
        user_display_name = CONFIG["USER_INFO"].get("real_name")
        generate_dynamic_report(df, user_display_name, CONFIG)
        create_visualizations(df, user_display_name, CONFIG)
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

if __name__ == '__main__':
    main()