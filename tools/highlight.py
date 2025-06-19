from datetime import datetime
import dateutil
import pandas as pd
import random
import re
from collections import Counter

class Highlighter:

    def __init__(self, df: pd.DataFrame, config: dict, recorpus: dict):
        
        self.__df = df
        self.__config = config
        self.earned_achievements = {}
        self.__tags = recorpus["HIGHLIGHTS"]["TAGS"]
        if self.__df.empty:
            return
        self.__pre_calcualte()

    def __pre_calcualte(self):
        #  --- 预计算常用数据 ---
        self.__df_with_mutual = self.__df[self.__df['has_mutual_test'] == True]
        self.__total_mutual_opportunities = len(self.__df_with_mutual)
        self.__mutual_df = self.__df_with_mutual.dropna(subset=['room_level'])
        
        self.__a_rate, self.__b_rate, self.__c_rate = 0, 0, 0
        if self.__total_mutual_opportunities > 0:
            room_counts = self.__mutual_df['room_level'].value_counts()
            self.__a_rate = room_counts.get('A', 0) / self.__total_mutual_opportunities
            self.__b_rate = room_counts.get('B', 0) / self.__total_mutual_opportunities
            self.__c_rate = room_counts.get('C', 0) / self.__total_mutual_opportunities

        self.__total_hacks = self.__mutual_df['hack_success'].sum()
        self.__total_hacked = self.__mutual_df['hacked_success'].sum()
        self.__total_hacks_attempted = self.__df['hack_total_attempts'].sum()
        
        self.__total_forum_activity = self.__df['essential_posts_authored'].sum() + \
                                      self.__df['official_replies'].sum() + \
                                      self.__df['peer_assists'].sum()

        self.__strong_scores = self.__df['strong_test_score'].dropna()
        self.__submit_times_df = self.__df.dropna(subset=['public_test_used_times'])


    def generate(self):
        if self.__df.empty:
            return [], {}

        for highlight in self.__tags.keys():
            func = f"_is_{highlight.lower()}"
            getattr(self, func)()

        # --- 随机选择5个亮点用于报告主体展示 ---
        if not self.earned_achievements:
            return [], {}

        highlight_texts_to_choose_from = [d['description'] for d in self.earned_achievements.values()]
        random.shuffle(highlight_texts_to_choose_from)
        final_highlights_for_display = highlight_texts_to_choose_from[:5]

        return final_highlights_for_display, self.earned_achievements

    # --- 辅助方法 ---

    def _add_highlight(self, key: str, context_str: str, **kwargs):
        if key not in self.earned_achievements:
            template = self.__tags[key]["description"]
            self.earned_achievements[key] = {
                'description': template.format(**kwargs),
                'context': context_str
            }
            
    def __is_target(self, data_dict):
        if not isinstance(data_dict, dict): return False
        user_id = str(self.__config["USER_INFO"]["student_id"])
        if 'student_id' in data_dict and str(data_dict['student_id']) == user_id:
            return True
        return any(data_dict.get(k) == v for k, v in self.__config["USER_INFO"].items() if v is not None)

    # --- 单个成就判断方法 (全局/学期) ---

    def _is_thousand_blade_wolf_king(self):
        if self.__total_hacks_attempted > 0:
            success_rate = (self.__total_hacks / self.__total_hacks_attempted) * 100
            if self.__total_hacks_attempted > 500 and success_rate > 5:
                self._add_highlight(
                    "THOUSAND_BLADE_WOLF_KING", "于 整个学期",
                    total_attempts=int(self.__total_hacks_attempted),
                    total_hacks=int(self.__total_hacks),
                    rate=success_rate
                )

    def _is_a_room_regular(self):
        if self.__total_mutual_opportunities > 0 and self.__a_rate >= 0.75:
            self._add_highlight("A_ROOM_REGULAR", "于 整个学期")

    def _is_glass_cannon(self):
        if self.__total_hacks > 15 and self.__total_hacked > 10:
            self._add_highlight("GLASS_CANNON", "于 整个学期")

    def _is_lone_swordsman(self):
        avg_score = self.__strong_scores.mean() if not self.__strong_scores.empty else 0
        if self.__total_hacks <= 2 and self.__total_hacked <= 2 and avg_score > 85:
            self._add_highlight("LONE_SWORDSMAN", "于 整个学期")

    def _is_roller_coaster_rider(self):
        if self.__total_mutual_opportunities > 2 and max(self.__a_rate, self.__b_rate, self.__c_rate) < 0.6:
            self._add_highlight("ROLLER_COASTER_RIDER", "于 整个学期")

    def _is_golden_silence(self):
        if self.__total_forum_activity <= 1:
            self._add_highlight("GOLDEN_SILENCE", "于 整个学期")

    def _is_community_pillar(self):
        essential_posts_df = self.__df[self.__df['essential_posts_authored'] > 0]
        if not essential_posts_df.empty:
            first_essential_title = essential_posts_df.iloc[0]['essential_post_titles'][0]
            self._add_highlight("COMMUNITY_PILLAR", f"于《{first_essential_title}》", post_title=first_essential_title)

    def _is_rigorous_inquirer(self):
        if self.__df['official_replies'].sum() > 0:
            self._add_highlight("RIGOROUS_INQUIRER", "于 整个学期")

    def _is_peer_mentor(self):
        peer_assists_df = self.__df[self.__df['peer_assists'] > 0]
        if not peer_assists_df.empty:
            first_assist_title = peer_assists_df.iloc[0]['assisted_post_titles'][0]
            self._add_highlight("PEER_MENTOR", f"于《{first_assist_title}》", post_title=first_assist_title)

    def _is_foundation_builder(self):
        if not self.__strong_scores.empty and self.__strong_scores.mean() < 75 and len(self.__df) > 10:
            worst_hw = self.__df.loc[self.__strong_scores.idxmin()]
            self._add_highlight("FOUNDATION_BUILDER", f"于 {worst_hw['name']}", hw_name=worst_hw['name'])

    def _is_diligent_explorer(self):
        if not self.__submit_times_df.empty:
            total_submissions = int(self.__submit_times_df['public_test_used_times'].sum())
            if total_submissions > 30:
                self._add_highlight("DILIGENT_EXPLORER", "于 整个学期", total_submissions=total_submissions)
    
    def _is_the_perseverer(self):
        if len(self.__strong_scores) > 1:
            for i in range(len(self.__df) - 1):
                hw1, hw2 = self.__df.iloc[i], self.__df.iloc[i+1]
                s1, s2 = hw1.get('strong_test_score'), hw2.get('strong_test_score')
                if pd.notna(s1) and pd.notna(s2) and s1 < 80 and s2 - s1 > 15:
                    self._add_highlight("THE_PERSEVERER", f"于 {hw2['name']}", low_score_hw=hw1['name'], rebound_hw=hw2['name'])
                    return # 只记录第一次

    def _is_bug_fixer_pro(self):
        bugfix_df = self.__df.dropna(subset=['bug_fix_hacked_count'])
        if not bugfix_df.empty and bugfix_df['bug_fix_hacked_count'].sum() > 0 and bugfix_df['bug_fix_unfixed_count'].sum() == 0:
            self._add_highlight("BUG_FIXER_PRO", "于 整个学期")

    def _is_performance_challenger(self):
        unit2_df = self.__df[self.__df['unit'].str.contains("第二单元", na=False)]
        if not unit2_df.empty:
            has_perf_issues = any("TIME" in str(s) for s in unit2_df['strong_test_issues'].dropna())
            if has_perf_issues and unit2_df['strong_test_score'].mean() > 95:
                self._add_highlight("PERFORMANCE_CHALLENGER", "于 第二单元")
    
    def _is_rock_solid(self):
        if not self.__strong_scores.empty and not self.__mutual_df.empty and self.__strong_scores.min() > 95 and self.__mutual_df['hacked_success'].sum() <= 1:
            self._add_highlight("ROCK_SOLID", "于 整个学期", min_score=self.__strong_scores.min())

    def _is_defense_master(self):
        if not self.__mutual_df.empty and (self.__mutual_df['hacked_success'] == 0).mean() >= 0.75:
            self._add_highlight("DEFENSE_MASTER", "于 整个学期")

    def _is_top_scorer(self):
        if not self.__strong_scores.empty and self.__strong_scores.mean() > 98.5:
            self._add_highlight("TOP_SCORER", "于 整个学期", avg_score=self.__strong_scores.mean())
            
    def _is_code_artisan(self):
        style_df = self.__df.dropna(subset=['style_score'])
        if not style_df.empty and (style_df['style_score'] == 100).all():
            self._add_highlight("CODE_ARTISAN", "于 整个学期")

    def _is_score_actuary(self):
        bugfix_df = self.__df.dropna(subset=['bug_fix_hack_score', 'bug_fix_hacked_score'])
        if not bugfix_df.empty:
            total_hack_score = bugfix_df['bug_fix_hack_score'].sum()
            total_hacked_score = bugfix_df['bug_fix_hacked_score'].sum()
            if total_hacked_score > 0 and total_hack_score > 0:
                ratio = total_hack_score / total_hacked_score
                if 0.9 <= ratio <= 1.1:
                    self._add_highlight("SCORE_ACTUARY", "于 整个学期")
                    
    def _is_proactive_refactorer(self):
        total_refactors = self.__df['commit_refactor_count'].sum()
        if total_refactors >= 3:
            self._add_highlight("PROACTIVE_REFACTORER", "于 整个学期", count=int(total_refactors))

    def _is_version_control_guru(self):
        total_version_commits = self.__df['commit_keywords'].apply(lambda x: x.get('versioning', 0)).sum()
        if total_version_commits >= 5:
            self._add_highlight("VERSION_CONTROL_GURU", "于 整个学期", count=int(total_version_commits))

    def _is_night_owl_coder(self):
        night_commits, total_commits = 0, 0
        for commits in self.__df['commits']:
            if commits:
                total_commits += len(commits)
                for commit in commits:
                    if 1 <= commit['timestamp'].hour <= 4:
                        night_commits += 1
        if total_commits > 0 and (night_commits / total_commits) > 0.3:
            percentage = int((night_commits / total_commits) * 100)
            self._add_highlight("NIGHT_OWL_CODER", "于 整个学期", percentage=percentage)
    
    def _is_nemesis(self):
        hacked_targets_counter = Counter()

        for _, hw in self.__df.iterrows():
            my_successful_events = hw.get('mutual_test_events', [])
            # print(my_successful_events)
            if not my_successful_events:
                continue
            for event in my_successful_events:
                hacked_info = event.get('hacked', {})
                if isinstance(hacked_info, dict):
                    target_id = hacked_info.get('student_id')
                    if target_id:
                        hacked_targets_counter[target_id] += 1
        
        if hacked_targets_counter:
            top_nemesis = hacked_targets_counter.most_common(1)[0]
            nemesis_id, max_count = top_nemesis

            if max_count > 3:
                self._add_highlight(
                    "NEMESIS",
                    "于 整个学期",  # 成就上下文
                    count=int(max_count)
                )

    def _is_knowledge_sharer(self):
        total_free_discuss_posts = self.__df['free_discuss_posts_authored'].sum()
        
        if total_free_discuss_posts >= 1:
            first_share_hw = self.__df[self.__df['free_discuss_posts_authored'] > 0].iloc[0]
            
            self._add_highlight(
                "KNOWLEDGE_SHARER",
                f"于 {first_share_hw['name']}", # 上下文
                hw_name=first_share_hw['name'],
                count=int(total_free_discuss_posts)
            )

    def _is_proactive_explorer(self):

        total_issue_posts = self.__df['issue_posts_authored'].sum()
        total_peer_assists = self.__df['peer_assists'].sum()
        
        total_proactive_actions = total_issue_posts + total_peer_assists
        
        if total_proactive_actions > 2:
            self._add_highlight(
                "PROACTIVE_EXPLORER",
                "于 整个学期", # 上下文
                count=int(total_proactive_actions)
            )

    # --- 单个成就判断方法 (作业级，内部循环) ---

    def _is_active_collaborator(self):
        if not self.__mutual_df.empty and 'hack_total_attempts' in self.__mutual_df.columns:
             active_row = self.__mutual_df.loc[self.__mutual_df['hack_total_attempts'].idxmax(skipna=True)]
             if active_row is not None and active_row['hack_total_attempts'] > 10:
                self._add_highlight("ACTIVE_COLLABORATOR", f"于 {active_row['name']}", hw_name=active_row['name'], hack_attempts=int(active_row['hack_total_attempts']))

    def _is_deadline_comeback(self):
        ddl_comeback_df = self.__df[(self.__df['start_ratio'] > 0.8) & (self.__df['strong_test_score'] > 85)]
        if len(ddl_comeback_df) >= 2:
            hw = ddl_comeback_df.iloc[0]
            self._add_highlight("DEADLINE_COMEBACK", f"于 {hw['name']}", hw_name=hw['name'])

    def _is_efficiency_ace(self):
        if not self.__submit_times_df.empty:
            min_submit_row = self.__submit_times_df.loc[self.__submit_times_df['public_test_used_times'].idxmin()]
            if min_submit_row['public_test_used_times'] <= 2:
                self._add_highlight("EFFICIENCY_ACE", f"于 {min_submit_row['name']}", hw_name=min_submit_row['name'])

    def _is_fast_starter(self):
        early_submitters = self.__df[self.__df['start_ratio'] < 0.1]
        if len(early_submitters) >= 3:
            hw = early_submitters.iloc[0]
            self._add_highlight("FAST_STARTER", f"于 {hw['name']}", hw_name=hw['name'])

    def _is_hack_artist(self):
        if not self.__mutual_df.empty and not self.__mutual_df['hack_success'].empty:
            max_hack_row = self.__mutual_df.loc[self.__mutual_df['hack_success'].idxmax()]
            if max_hack_row['hack_success'] >= 10:
                self._add_highlight("HACK_ARTIST", f"于 {max_hack_row['name']}", hw_name=max_hack_row['name'], count=int(max_hack_row['hack_success']))

    def _is_coding_marathoner(self):
        if self.__df['commit_count'].max() > 20:
            marathon_hw = self.__df.loc[self.__df['commit_count'].idxmax()]
            self._add_highlight("CODING_MARATHONER", f"于 {marathon_hw['name']}", hw_name=marathon_hw['name'], count=int(marathon_hw['commit_count']))

    def _is_public_enemy_no_1(self):
        for _, hw in self.__df.iterrows():
            room_total_hacked = hw.get('room_total_hacked', 0)
            member_count = hw.get('room_member_count', 0)
            if room_total_hacked > 0 and member_count > 1 and \
               (hw.get('hacked_success', 0) / room_total_hacked) > 0.4 and \
               hw.get('hacked_total_attempts', 0) > (member_count / 2):
                self._add_highlight("PUBLIC_ENEMY_NO_1", f"于 {hw['name']}", hw_name=hw['name'])
                return

    def _is_performance_ace(self):
        for _, hw in self.__df.iterrows():
            if hw['unit'].startswith("第一单元") or hw['unit'].startswith("第二单元"):
                details = hw.get('strong_test_details', [])
                if details:
                    scores = [r.get('score', 0) for r in details]
                    if scores and all(s > 98 for s in scores):
                        self._add_highlight("PERFORMANCE_ACE", f"于 {hw['name']}", hw_name=hw['name'])
                        return

    def _is_correctness_first(self):
        for _, hw in self.__df.iterrows():
            if hw['unit'].startswith("第一单元") or hw['unit'].startswith("第二单元"):
                details = hw.get('strong_test_details', [])
                if details:
                    scores = [r.get('score', 0) for r in details]
                    if scores and all(85 <= s <= 92 for s in scores):
                        self._add_highlight("CORRECTNESS_FIRST", f"于 {hw['name']}", hw_name=hw['name'])
                        return

    def _is_performance_gambler(self):
        for _, hw in self.__df.iterrows():
            if hw['unit'].startswith("第一单元") or hw['unit'].startswith("第二单元"):
                details = hw.get('strong_test_details', [])
                if details:
                    scores = [r.get('score', 0) for r in details]
                    if scores and any(s == 0 for s in scores) and any(s > 98 for s in scores):
                        self._add_highlight("PERFORMANCE_GAMBLER", f"于 {hw['name']}", hw_name=hw['name'])
                        return
    
    def _is_annihilation(self):
        for _, hw in self.__df.iterrows():
            member_count = hw.get('room_member_count', 0)
            target_count = hw.get('successful_hack_targets', 0)
            if member_count > 1 and target_count == (member_count - 1):
                self._add_highlight("ANNIHILATION", f"于 {hw['name']}", hw_name=hw['name'])
                return
    
    def _is_phoenix_rebirth(self):
        for _, hw in self.__df.iterrows():
            if hw.get('room_total_hacked', 0) > 25 and hw.get('hacked_success', 0) > 0 and hw.get('strong_test_score', 0) > 90:
                self._add_highlight("PHOENIX_REBIRTH", f"于 {hw['name']}", hw_name=hw['name'])
                return

    def _is_ice_breaker(self):
        for _, hw in self.__df.iterrows():
            room_hack_success = hw.get('room_total_hack_success', 0)
            my_hack_success = hw.get('hack_success', 0)
            if 0 < room_hack_success <= 5 and my_hack_success > 0 and (my_hack_success / room_hack_success) >= 0.8:
                self._add_highlight("ICE_BREAKER", f"于 {hw['name']}", hw_name=hw['name'])
                return

    def _is_buzzer_beater(self):
        for _, hw in self.__df.iterrows():
            my_events = hw.get('mutual_test_events', [])
            end_time = hw.get('mutual_test_end_time')
            if my_events and pd.notna(end_time):
                for event in my_events:
                    hack_time = pd.to_datetime(event.get('submitted_at'), errors='coerce')
                    if pd.notna(hack_time) and (end_time - hack_time).total_seconds() < 3600:
                        self._add_highlight("BUZZER_BEATER", f"于 {hw['name']}", hw_name=hw['name'])
                        return

    def _is_chain_reaction(self):
        for _, hw in self.__df.iterrows():
            my_events = hw.get('mutual_test_events', [])
            if my_events:
                submission_counts = Counter(e.get('submitted_at') for e in my_events if e.get('submitted_at'))
                if submission_counts:
                    top_submission = submission_counts.most_common(1)[0]
                    if top_submission[1] >= 3:
                        self._add_highlight("CHAIN_REACTION", f"于 {hw['name']}", hw_name=hw['name'], count=top_submission[1])
                        return

    def _is_counter_attack(self):
        for _, hw in self.__df.iterrows():
            all_events = hw.get('room_events', [])
            if all_events:
                was_hacked = False
                hacked_time = pd.Timestamp.min 
                
                # Replicate the buggy string sort from origin.py
                sorted_events = sorted(all_events, key=lambda x: x.get('submitted_at', ''))
                
                for event in sorted_events:
                    event_time = pd.to_datetime(event.get('submitted_at'), errors='coerce')
                    if pd.isna(event_time): continue
                    
                    if 'hacked' in event and self.__is_target(event.get('hacked', {})):
                        was_hacked = True
                        hacked_time = event_time
                    
                    if was_hacked and 'hack' in event and self.__is_target(event.get('hack', {})) and event_time > hacked_time:
                        self._add_highlight("COUNTER_ATTACK", f"于 {hw['name']}", hw_name=hw['name'])
                        return
                    
    def _is_first_blood(self):
        for _, hw in self.__df.iterrows():
            all_events = hw.get('room_events', [])
            if not all_events:
                continue
            sorted_events = sorted(all_events, key=lambda x: x.get('submitted_at', ''))

            if sorted_events:
                first_event = sorted_events[0]
                print(first_event)
                if 'hack' in first_event and self.__is_target(first_event.get('hack', {})):
                    self._add_highlight("FIRST_BLOOD", f"于 {hw['name']}", hw_name=hw['name'])
                    return
    
    def _is_planning_master(self):
        for _, hw in self.__df.iterrows():
            if (hw.get('ddl_index', 1) < 0.3 and hw.get('public_test_used_times', 99) <= 2 and
                hw.get('strong_test_score', 0) == 100 and hw.get('hacked_success', 99) == 0):
                self._add_highlight("PLANNING_MASTER", f"于 {hw['name']}", hw_name=hw['name'], count=int(hw['public_test_used_times']))
                return
    
    def _is_iron_wall_squad(self):
        for _, hw in self.__df.iterrows():
            if (hw.get('room_total_hack_success', 99) == 0 and hw.get('room_total_hack_attempts', 0) > 50):
                self._add_highlight("IRON_WALL_SQUAD", f"于 {hw['name']}", hw_name=hw['name'], total_attacks=int(hw['room_total_hack_attempts']))
                return
    
    def _is_precision_striker(self):
        for _, hw in self.__df.iterrows():
            if pd.notna(hw.get('hack_success_rate')) and hw['hack_success_rate'] > 8:
                self._add_highlight("PRECISION_STRIKER", f"于 {hw['name']}", hw_name=hw['name'], rate=hw['hack_success_rate'])
                return

    def _is_tactical_master(self):
        for _, hw in self.__df.iterrows():
            targets = hw.get('successful_hack_targets', 0)
            successes = hw.get('hack_success', 0)
            if targets > 0 and successes > 3 and (successes / targets) > 1.8:
                self._add_highlight("TACTICAL_MASTER", f"于 {hw['name']}", hw_name=hw['name'], target_count=int(targets), hack_count=int(successes))
                return

    def _is_storm_survivor(self):
        for _, hw in self.__df.iterrows():
            if hw.get('room_total_hacked', 0) > 20 and hw.get('hacked_success', 100) <= 1:
                self._add_highlight("STORM_SURVIVOR", f"于 {hw['name']}", hw_name=hw['name'], room_total_hacked=int(hw.get('room_total_hacked', 0)), self_hacked=int(hw.get('hacked_success', 0)))
                return

    # --- 单个成就判断方法 (单元级) ---
    
    def _is_expression_guru(self):
        unit1_df = self.__df[self.__df['unit'].str.contains("第一单元", na=False)]
        if not unit1_df.empty and unit1_df['strong_test_score'].mean() > 98 and unit1_df['hacked_success'].sum() <= 4:
            self._add_highlight("EXPRESSION_GURU", "于 第一单元")

    def _is_concurrency_conductor(self):
        unit2_df = self.__df[self.__df['unit'].str.contains("第二单元", na=False)]
        if not unit2_df.empty and unit2_df['strong_test_score'].mean() > 95 and unit2_df['hacked_success'].sum() <= 8:
            self._add_highlight("CONCURRENCY_CONDUCTOR", "于 第二单元")
    
    def _is_jml_master(self):
        unit3_df = self.__df[self.__df['unit'].str.contains("第三单元", na=False)]
        if not unit3_df.empty and unit3_df['strong_test_score'].mean() > 99 and unit3_df['hacked_success'].sum() == 0:
            self._add_highlight("JML_MASTER", "于 第三单元")

    def _is_uml_expert(self):
        unit4_df = self.__df[self.__df['unit'].str.contains("第四单元", na=False)]
        if not unit4_df.empty and unit4_df['strong_test_score'].mean() == 100:
            is_perfect = all(
                all(r['message'] == 'ACCEPTED' for r in row.get('uml_detailed_results', []))
                for _, row in unit4_df.iterrows() if row.get('uml_detailed_results')
            )
            if is_perfect:
                self._add_highlight("UML_EXPERT", "于 第四单元")

    def _is_refactor_virtuoso(self):
        for unit_name in self.__df['unit'].unique():
            unit_df = self.__df[self.__df['unit'] == unit_name].sort_values('hw_num')
            if len(unit_df) > 1:
                scores = unit_df['strong_test_score'].dropna()
                if len(scores) > 1 and scores.iloc[-1] - scores.iloc[0] > 10 and scores.iloc[-1] > 95:
                    unit_name_short = re.sub(r'：.*', '', unit_name)
                    self._add_highlight("REFACTOR_VIRTUOSO", f"于 {unit_name_short}", 
                                      unit_name=unit_name_short, 
                                      hw_name_before=unit_df.iloc[0]['name'], 
                                      hw_name_after=unit_df.iloc[-1]['name'])
                    return # 每个单元只记录一次

    def _is_comeback_king(self):
        unit_scores = self.__df.groupby('unit')['strong_test_score'].mean()
        u1_key, u4_key = next((k for k in unit_scores.index if "第一单元" in k), None), \
                         next((k for k in unit_scores.index if "第四单元" in k), None)
        if u1_key and u4_key:
            u1_score, u4_score = unit_scores.get(u1_key), unit_scores.get(u4_key)
            if pd.notna(u1_score) and pd.notna(u4_score) and u4_score > u1_score + 2:
                self._add_highlight("COMEBACK_KING", "于 整个学期", u1_score=u1_score, u4_score=u4_score)

    # --- 元成就判断 ---

    def _is_decorated_developer(self):
        if len(self.earned_achievements) > 5:
            self._add_highlight("DECORATED_DEVELOPER", "于 整个学期")

    def _is_collection(self):
        all_possible_keys = set(self.__tags.keys())
        keys_to_check = all_possible_keys - {"DECORATED_DEVELOPER", "COLLECTION"} # 避免元成就自包含
        # 如果已经解锁的成就数量，达到了总可能成就的70%
        if len(self.earned_achievements) >= len(keys_to_check) * 0.7:
            self._add_highlight("COLLECTION", "于 整个学期")