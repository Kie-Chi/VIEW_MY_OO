
from datetime import timedelta, datetime, timezone
import json
import os
from visualize import Visualizer
from highlight import Highlighter
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from collections import Counter
import sys
import traceback
import yaml

# --- Config  ---
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
# --- end of Config ---



# --- matplotlib ---
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# --- end of matplotlib ---



# --- Corpus ---
REPORT_CORPUS = {}
# --- end of Corpus ---



# --- exception catch ---
def exit_with_error(message: str):
    """打印一条致命错误信息到 stderr 并以状态码 1 退出脚本。"""
    print(f"\n[CRITICAL ERROR] {message}", file=sys.stderr)
    print("[INFO] Script terminated due to a critical error.", file=sys.stderr)
    sys.exit(1)
# --- end of exception ---



# --- process and enrich DataFrame from preprocess.py
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

    # --- 步骤 1: 计算第一层衍生指标 (许多来自 origin.py 的 parse_course_data) ---
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
    with np.errstate(divide='ignore', invalid='ignore'):
        df['room_avg_hacked'] = df['room_total_hacked'] / df['room_member_count']
    df['room_avg_hacked'] = df['room_avg_hacked'].fillna(0)


    # 个人互测统计
    def get_my_mutual_stats(row):
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
    
    def count_unique_targets(my_hack_events):
        if not my_hack_events: return 0
        return len(set(e['hacked']['student_id'] for e in my_hack_events if 'hacked' in e and 'student_id' in e['hacked']))
    df['successful_hack_targets'] = df['mutual_test_events'].apply(count_unique_targets)

    # Bug修复相关衍生 (从嵌套字典中提取)
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
    
    # 步骤 5, 对讨论区的处理
    if 'forum_activities' in df.columns:
        def count_essentials(activities):
            return sum(1 for act in activities if act.get('type') == 'authored' and act.get('priority') == 'essential')
        
        def get_essential_titles(activities):
            return [act['title'] for act in activities if act.get('type') == 'authored' and act.get('priority') == 'essential']

        def count_official_replies(activities):
            return sum(1 for act in activities if act.get('type') == 'commented' and act.get('post_priority') == 'top')

        def count_peer_assists(activities):
            user_name = config["USER_INFO"]["real_name"]
            return sum(1 for act in activities if act.get('type') == 'commented' and act.get('post_category') == 'issue' and act.get('post_author') != user_name)

        def get_assisted_titles(activities):
             user_name = config["USER_INFO"]["real_name"]
             return [act['post_title'] for act in activities if act.get('type') == 'commented' and act.get('post_category') == 'issue' and act.get('post_author') != user_name]
        
        # 新增：知识布道者和先行探索者需要的列
        def count_free_discuss_posts(activities):
            return sum(1 for act in activities if act.get('type') == 'authored' and act.get('category') == 'free_discuss')

        def count_issue_posts(activities):
            return sum(1 for act in activities if act.get('type') == 'authored' and act.get('category') == 'issue')

        df['essential_posts_authored'] = df['forum_activities'].apply(count_essentials)
        df['essential_post_titles'] = df['forum_activities'].apply(get_essential_titles)
        df['official_replies'] = df['forum_activities'].apply(count_official_replies)
        df['peer_assists'] = df['forum_activities'].apply(count_peer_assists)
        df['assisted_post_titles'] = df['forum_activities'].apply(get_assisted_titles)
        df['free_discuss_posts_authored'] = df['forum_activities'].apply(count_free_discuss_posts)
        df['issue_posts_authored'] = df['forum_activities'].apply(count_issue_posts)

    # --- 步骤 5: 清理不再需要的原始数据列 ---
    df = df.drop(columns=['room_members'], errors='ignore')
    df = df.drop(columns=['forum_activities'], errors='ignore')
    return df
# --- end of enrich ---


# --- Analyze ---
class Analyzer:
    def __init__(self, df, config):
        self.__df = df
        self.__config = config
        self.__highlighter = Highlighter(
            self.__df,
            self.__config,
            REPORT_CORPUS
        )

    def __identify(self):
        if self.__df.empty: return "BALANCED"
        
        strong_scores = self.__df['strong_test_score'].dropna()
        avg_score = strong_scores.mean() if not strong_scores.empty else 100
        if avg_score < 75:
            return "BALANCED_GENTLE"

        mutual_df = self.__df[self.__df.get('has_mutual_test', pd.Series(False))]
        if 'start_ratio' in self.__df.columns and self.__df['start_ratio'].dropna().mean() > 0.7: return "SPRINTER"
        if not mutual_df.empty and mutual_df['hack_success'].sum() > 25: return "HUNTER"
        if (not mutual_df.empty and mutual_df['hacked_success'].sum() <= 3) and self.__df['strong_test_score'].var() < 10: return "FORTRESS"
        if self.__df['public_test_used_times'].dropna().mean() > 6: return "GRINDER"
        return "BALANCED"

    def __format_uml(self, hw_row):
        uml_results = hw_row.get('uml_detailed_results', [])
        if not uml_results: return ""
        failed_checks = [r['name'] for r in uml_results if r['message'] != 'ACCEPTED']
        if not failed_checks: return random.choice(REPORT_CORPUS["UML"]["PERFECT"])
        else: return random.choice(REPORT_CORPUS["UML"]["IMPERFECT"]).format(issues=', '.join(failed_checks))

    def __analyze_overall(self):
        analysis_texts = []
        
        strong_scores = self.__df['strong_test_score'].dropna()
        avg_score = strong_scores.mean() if not strong_scores.empty else 0

        if not strong_scores.empty:
            var_score = strong_scores.var()
            analysis_texts.append(f"强测表现: 平均分 {avg_score:.2f} | 稳定性 (方差) {var_score:.2f}")
            
            # --- [V8.7 改造] 强测表现分析优化 ---
            if avg_score > 98:
                analysis_texts.append(random.choice(REPORT_CORPUS["ANALYSIS"]["STRONG_TEST"]["HIGH_SCORE"]))
            elif avg_score < 75:
                struggle_hws = self.__df[self.__df['strong_test_score'] < 70]['name'].tolist()
                if struggle_hws:
                    analysis_texts.append(random.choice(REPORT_CORPUS["ANALYSIS"]["STRONG_TEST"]["STRUGGLE"]).format(hw_names=', '.join(struggle_hws)))
            else:
                imperfect_hws = self.__df[self.__df['strong_test_score'] < 100]['name'].tolist()
                if imperfect_hws:
                    analysis_texts.append(random.choice(REPORT_CORPUS["ANALYSIS"]["STRONG_TEST"]["IMPERFECTION"]).format(hw_names=', '.join(imperfect_hws[:2])))
            
            if var_score < 15:
                if avg_score > 80: # 1. 高分稳定 -> 赞美
                    analysis_texts.append(random.choice(REPORT_CORPUS["ANALYSIS"]["CONSISTENCY"]["STABLE"]).format(variance=var_score))
                else: # 2. 低分稳定 -> 鼓励寻求突破 (调用新增语料库)
                    analysis_texts.append(random.choice(REPORT_CORPUS["ANALYSIS"]["CONSISTENCY"]["LOW_SCORE_STABLE"]).format(variance=var_score))
            else:
                best_hw = self.__df.loc[self.__df['strong_test_score'].idxmax()]['name'] if pd.notna(self.__df['strong_test_score'].max()) else '某次作业'
                worst_hw = self.__df.loc[self.__df['strong_test_score'].idxmin()]['name'] if pd.notna(self.__df['strong_test_score'].min()) else '另一次作业'
                analysis_texts.append(random.choice(REPORT_CORPUS["ANALYSIS"]["CONSISTENCY"]["VOLATILE"]).format(variance=var_score, best_hw=best_hw, worst_hw=worst_hw))

        mutual_df = self.__df[self.__df.get('has_mutual_test', pd.Series(False))].dropna(subset=['hack_success', 'hacked_success', 'hacked_total_attempts'])
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
            room_df = self.__df.dropna(subset=['room_level'])
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

        unit2_df = self.__df[self.__df['unit'].str.contains("第二单元", na=False)]
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
        commit_df = self.__df[self.__df['commit_count'] > 2].copy()
        
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

    def __analyze_hack(self):
        texts = []
        # 筛选出包含有效互测事件和相关时间的作业
        mutual_df = self.__df.dropna(subset=['mutual_test_start_time', 'mutual_test_end_time', 'hack_success_rate'])
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

    def __format(self, hw_row):
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

    def __generate_prompt(self):
        user_name = self.__config.get("USER_INFO", {"real_name": "007"}).get("real_name")
        print("\n" + "="*80)
        print(f" {user_name} - OO课程动态学习轨迹报告".center(80, "="))
        print("="*80)

        if self.__df.empty:
            print("\n未找到该学生的有效作业数据，请检查配置文件。")
            raise ValueError(f"Can't find valid infomation about the User {user_name}")
        persona_key = self.__identify()
        persona_text = REPORT_CORPUS["PERSONA"].get(persona_key, REPORT_CORPUS["PERSONA"]["BALANCED"])
        print("\n" + persona_text.format(user_name=user_name))

    def __generate_highlights(self):
        highlights_for_display, earned_achievements_details = self.__highlighter.generate()

        if highlights_for_display:
            print("\n" + "--- 1. 个人亮点标签 ---".center(70))
            print(random.choice(REPORT_CORPUS["HIGHLIGHTS"]["INTRO"]))
            for tag_text in highlights_for_display:
                print(tag_text)
        
        return earned_achievements_details

    def __generate_overall(self):
        print("\n" + "--- 2. 宏观学期表现与深度洞察 ---".center(70))
        for text in self.__analyze_overall():
            print(text)

    def __generate_bugfix(self):
        print("\n" + "--- 3. 开发者责任感与调试能力 (Bug修复) ---".center(70))
        bugfix_df = self.__df.dropna(subset=['bug_fix_hacked_count'])
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

    def __generate_trends(self):
        print("\n" + "--- 4. 单元深度与成长轨迹 ---".center(70))
        unit_paradigms = {"第一单元": "递归下降", "第二单元": "多线程", "第三单元": "JML规格", "第四单元": "UML解析"}
        for unit_name_full, hw_nums in self.__config["UNIT_MAP"].items():
            unit_df = self.__df[self.__df['unit'] == unit_name_full]
            unit_name_short = re.sub(r'：.*', '', unit_name_full)
            if not unit_df.empty and pd.notna(unit_df['strong_test_score'].mean()):
                print(random.choice(REPORT_CORPUS["ANALYSIS"]["UNIT"]).format(unit_name=unit_name_short, avg_score=unit_df['strong_test_score'].mean(), unit_paradigm=unit_paradigms.get(unit_name_short, "核心技术"), hacks=int(unit_df['hack_success'].sum()), hacked=int(unit_df['hacked_success'].sum())))

        strong_scores = self.__df['strong_test_score'].dropna()
        if len(strong_scores) > 8:
            early_avg, later_avg = strong_scores.iloc[:len(strong_scores)//2].mean(), strong_scores.iloc[len(strong_scores)//2:].mean()
            if later_avg > early_avg + 1:
                print(random.choice(REPORT_CORPUS["ANALYSIS"]["GROWTH"]).format(early_avg=early_avg, later_avg=later_avg))

    def __generate_risks(self):
        print("\n" + "--- 5. 提交行为与风险分析 ---".center(70))
        print(random.choice(REPORT_CORPUS["SUBMISSION"]["INTRO"]))
        submit_times_df = self.__df.dropna(subset=['public_test_used_times'])
        if not submit_times_df.empty and not submit_times_df['public_test_used_times'].empty:
            total_submissions = submit_times_df['public_test_used_times'].sum()
            print(f"本学期你共提交 {int(total_submissions)} 次代码。")
            most_submitted, least_submitted = submit_times_df.loc[submit_times_df['public_test_used_times'].idxmax()], submit_times_df.loc[submit_times_df['public_test_used_times'].idxmin()]
            print(random.choice(REPORT_CORPUS["SUBMISSION"]["MOST"]).format(hw_name=most_submitted['name'], count=int(most_submitted['public_test_used_times'])))
            print(random.choice(REPORT_CORPUS["SUBMISSION"]["LEAST"]).format(hw_name=least_submitted['name'], count=int(least_submitted['public_test_used_times'])))

        ddl_risk_df = self.__df[(self.__df['ddl_index'] > 0.9) & ((self.__df['strong_test_deduction_count'] > 0) | (self.__df['hacked_success'] > 0))]
        if len(ddl_risk_df) > len(self.__df) * 0.1:
            print(random.choice(REPORT_CORPUS["ANALYSIS"]["DDL"]))

    def __generate_forums(self):
        total_essentials = self.__df['essential_posts_authored'].sum()
        total_official_replies = self.__df['official_replies'].sum()
        total_peer_assists = self.__df['peer_assists'].sum()
        # 只有当学生有任何一种高质量互动时，才打印本章节
        if total_essentials > 0 or total_official_replies > 0 or total_peer_assists > 0:
            print("\n" + "--- [特别洞察] 社区互动与学习风格 ---".center(70))
            print(random.choice(REPORT_CORPUS["FORUM_ANALYSIS"]["INTRO"]))

            # 分析并打印“社区之光”行为
            if total_essentials > 0:
                # 安全地获取第一个精华帖的标题用于展示
                first_essential_title = self.__df[self.__df['essential_posts_authored'] > 0].iloc[0]['essential_post_titles'][0]
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

    def __generate_rooms(self):
        peace_room_text_generated = False
        for _, hw in self.__df.iterrows():
            if hw.get('room_total_hack_success', 99) == 0 and hw.get('room_total_hack_attempts', 0) > 50:
                if not peace_room_text_generated:
                    print("\n" + "--- [特别洞察] 房间生态分析 ---".center(70))
                    peace_room_text_generated = True
                print(random.choice(REPORT_CORPUS["MUTUAL_TEST"]["ROOM_ECOLOGY"]["PEACE_ROOM"]).format(
                    hw_name=hw['name'],
                    total_attacks=int(hw['room_total_hack_attempts'])
                ))

    def __generate_insights(self):
        hack_strategy_texts = self.__analyze_hack()
        if hack_strategy_texts:
            print("\n" + "--- 6. 洞察 ---".center(70))
            print(random.choice(REPORT_CORPUS["HACK_STRATEGY"]["INTRO"]))
            for text in hack_strategy_texts:
                print(text)

        self.__generate_forums()
        self.__generate_rooms()

    def __generate_homeworks(self):
        print("\n" + "--- 7. 逐次作业深度解析 ---".center(70))
        print(random.choice(REPORT_CORPUS["ANALYSIS"]["HW_INTRO"]))
        for _, hw in self.__df.iterrows():
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
                print(self.__format_uml(hw))
            print(f"  - {self.__format(hw)}")
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

    def __generate_achievements(self, details:dict):
        print("\n" + "--- 8. 个人成就墙 ---".center(70))

        all_achievements_data = REPORT_CORPUS["HIGHLIGHTS"]["TAGS"]
        total_achievements = len(all_achievements_data)
        completed_achievements = len(details.keys())
        
        print(f"成就进度：{completed_achievements} / {total_achievements}")

        unlocked_list = []
        locked_list = []

        for key, data in all_achievements_data.items():
            if key in details.keys():
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
                context_info = details[key].get('context', '')
                print(f"  {icon} {name} (达成于: {context_info.strip()})")

        # 打印未解锁成就
        if locked_list:
            print("\n--- 🔒 未解锁成就 ---")
            for key, data in locked_list:
                icon = data.get('icon', '❓')
                name = data.get('name', '未知成就')
                condition = data.get('condition', '未知条件')
                print(f"  {icon} {name} - {condition}")

    def __generate_conclusion(self):
        print("\n" + "="*80)
        print(" 学期旅程总结 ".center(80, "="))
        print("="*80)
        print(random.choice(REPORT_CORPUS["CONCLUSION"]))

    def generate_report(self):
        self.__generate_prompt()
        details = self.__generate_highlights()
        self.__generate_overall()
        self.__generate_bugfix()
        self.__generate_trends()
        self.__generate_risks()
        self.__generate_insights()
        self.__generate_homeworks()
        self.__generate_achievements(details)
        self.__generate_conclusion()

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
        Analyzer(df, CONFIG).generate_report()
        Visualizer(df, CONFIG).create_visualizations()
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