
import re
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class Visualizer:
    def __init__(self, df: pd.DataFrame, config:dict):
        self.__df = df
        self.__user = config.get("USER_INFO", {"real_name": "007"}).get("real_name")
        self.__unit_map = config.get("UNIT_MAP", {"1": [0], "2": [0], "3": [0], "4": [0]})

    def create_visualizations(self):
        print("\n正在生成可视化图表，请稍候...")
        try:
            self.__create_performance_dashboard()
        except Exception as e:
            print(f"[WARNING] 生成“综合表现仪表盘”失败，已跳过。错误: {e}", file=sys.stderr)
        try:
            self.__create_unit_radar_chart()
        except Exception as e:
            print(f"[WARNING] 生成“各单元能力雷达图”失败，已跳过。错误: {e}", file=sys.stderr)
        print("所有分析报告与图表已生成完毕！")

    def __create_performance_dashboard(self):
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(f'{self.__user} - OO课程综合表现仪表盘', fontsize=24, weight='bold')

        ax1 = axes[0, 0]
        df_strong = self.__df.dropna(subset=['strong_test_score'])
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
        analysis_df = self.__df.dropna(subset=['ddl_index', 'strong_test_deduction_count', 'hacked_success'])
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
        mutual_df = self.__df[self.__df.get('has_mutual_test', pd.Series(False))].dropna(subset=['offense_defense_ratio'])
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
        bugfix_df = self.__df.dropna(subset=['bug_fix_rate'])
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

    def __create_unit_radar_chart(self):
        unit_stats = {}
        for unit_name in self.__unit_map.keys():
            unit_df = self.__df[self.__df['unit'] == unit_name]
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
            ax.set_title(f'{self.__user} - 各单元能力雷达图', size=20, color='blue', y=1.1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.show()

    