# 数据接口文档

## 本文档描述了`analyze.py`可以分析的数据结构

## `user.info`
-   **文件名 (File Name):** `user.info`
-   **存储路径 (Location):** `.cache/user.info`
- **类型结构**：`json Object`
- **存储方式**：`json.dump()`
-   **用途 (Purpose):** 缓存从原始数据中识别出的目标用户信息，避免 `analyze.py` 重复执行耗时的用户搜索

### 字段说明

| 键 (Key)      | 类型 (Type) | 示例值                   | 说明                                                         |
|---------------|-------------|--------------------------|--------------------------------------------------------------|
| `student_id`  | `str`       | `"12345678"`             | 用户的学号，是核心的唯一标识符                             |
| `real_name`   | `str`       | `"某同学"`               | 用户的真实姓名         |
| `name`        | `str`       | `"某同学"`               | 姓名的别名或冗余字段，通常与 `real_name` 相同              |
| `email`       | `str`       | `"12345678@buaa.edu.cn"` | 用户的邮箱地址                                             |

---

## `tmp.pkl`
-   **文件名 (File Name):** `tmp.pkl`
-   **存储路径 (Location):** `.cache/tmp.pkl`
- **类型结构**：`pandas.DataFrame`
- **存储方式**：`pandas.DataFrame.topickle()`
-   **用途 (Purpose):** 缓存从原始数据中识别出的课程学习信息

### 主数据表结构

| #  | 列名 (Index Name)            | 数据类型 (Python Type)          | 示例值  | 说明、用途及分类值                                                                                 |
|:---|:---------------------------|:--------------------------------|:--------------------------------------|:---------------------------------------------------------------------------------------------------|
| 1  | `id`                         | `str`                           | `'606'`                               | **作业唯一ID**                                                                                   |
| 2  | `name`                       | `str`                           | `'第一次作业'`                        | **作业官方名称**，用于报告文本和图表标签                                                         |
| 3  | `has_mutual_test`            | `bool`                          | `True`                                | 指示作业是否有**互测**用于筛选数据                                                             |
| 4  | `mutual_test_start_time`     | `pandas.Timestamp` 或 `NaT`     | `YYYY-03-02 08:00:00`                 | **互测开始时间**，用于分析 Hack 时机                                                               |
| 5  | `mutual_test_end_time`       | `pandas.Timestamp` 或 `NaT`     | `YYYY-03-03 23:00:00`                 | **互测结束时间**，用于分析 Hack 时机                                                               |
| 6  | `room_members`               | `list[dict]`                    | `[...]`                               | **互测房间所有成员的关键信息**详情见 [附录 A](#a-room_members-结构)                             |
| 7  | `room_events`                | `list[dict]`                    | `[...]`                               | **互测房间所有成功 Hack 事件的日志**详情见 [附录 B](#b-room_events-结构)                          |
| 8  | `alias_name`                 | `str`                           | `'天枢星'`                            | **用户在互测中的别名**，用于报告文本                                                               |
| 9  | `room_level`                 | `str`                           | `'A'`                                 | **房间等级****分类值**: `'A'`, `'B'`, `'C'`用于计算加权防御扣分和成就判断                    |
| 10 | `bug_fix_details`            | `dict`                          | `{...}`                               | **Bug修复环节的个人统计**详情见 [附录 C](#c-bug_fix_details-结构)                             |
| 11 | `public_test_used_times`     | `int`                           | `2`                                   | **公测提交次数**，用于分析开发投入度和成就判断                                                   |
| 12 | `public_test_start_time`     | `pandas.Timestamp`              | `YYYY-02-27 04:00:00`                 | **公测开始时间**，用于计算开发启动和交付的相对时间点                                             |
| 13 | `public_test_end_time`       | `pandas.Timestamp`              | `YYYY-03-01 12:00:00`                 | **公测结束时间 (DDL)**，用于过滤 commit 记录和计算相对时间                                       |
| 14 | `public_test_last_submit`    | `pandas.Timestamp`              | `YYYY-02-27 22:18:09`                 | **最后一次公测提交时间**，用于计算 `ddl_index`                                                   |
| 15 | `strong_test_score`          | `float`                         | `97.7203`                             | **强测最终得分**，核心表现指标                                                                   |
| 16 | `strong_test_details`        | `list[dict]`                    | `[...]`                               | **强测详细结果列表**详情见 [附录 D](#d-strong_test_details-和-uml_detailed_results-结构)         |
| 17 | `style_score`                | `int` 或 `None`                 | `100`                                 | **代码风格分数**，用于“代码工匠”成就判断                                                         |
| 18 | `uml_detailed_results`       | `list[dict]`                    | `[...]`                               | **UML作业的详细检查结果**非UML作业此列为空详情见 [附录 D](#d-strong_test_details-和-uml_detailed_results-结构) |
| 19 | `hw_num`                     | `int`                           | `1`                                   | **作业数字编号**，用于排序和关联                                                                 |
| 20 | `commits`                    | `list[dict]`                    | `[...]`                               | **Git提交记录列表**详情见 [附录 E](#e-commits-结构)                                             |
| 21 | `forum_activities`           | `list[dict]`                    | `[...]`                               | **论坛活动列表**详情见 [附录 F](#f-forum_activities-结构)                                       |

---

### 附录：复杂类型结构详解 (仅保留 `analyze.py` 使用的字段)

#### A. `room_members` 结构
**类型结构**：`list[dict[Union...]]`
`room_members` 是一个列表，其中每个元素是一个**字典**，代表房间内的一名成员`analyze.py` 遍历此列表以计算个人和房间的互测指标

```json
{
    "student_id": "23371265",
    "hack": {
        "total": "42",
        "success": "0"
    },
    "hacked": {
        "total": "37",
        "success": "0"
    }
}
```
-   `student_id` (`str`): 用于识别目标用户
-   `hack.total` (`str`): 用户发起的 **Hack 总次数**
-   `hack.success` (`str`): 用户**成功 Hack 的次数**
-   `hacked.total` (`str`): 用户**被攻击的总次数**
-   `hacked.success` (`str`): 用户**被成功 Hack 的次数**

#### B. `room_events` 结构
**类型结构**：`list[dict[Union...]]`
`room_events` 是一个列表，其中每个元素是一个**字典**，代表一次成功的 Hack 事件

```json
{
    "submitted_at": "2025-03-10 20:48:08",
    "hack": {
        "student_id": "23371251"
    },
    "hacked": {
        "student_id": "23371464"
    }
}
```
-   `submitted_at` (`str`): 事件发生时间
-   `hack.student_id` (`str`): 攻击者ID，用于识别是否为目标用户发起的事件
-   `hacked.student_id` (`str`): 被攻击者ID，用于分析攻击目标的分布

#### C. `bug_fix_details` 结构
**类型结构**：`dict[Union...]`
`bug_fix_details` 是一个**字典**，包含用户在 Bug 修复环节的得分和计数

```json
{
    "hacked": {
        "score": 1.0,
        "count": 2,
        "unfixed": 0
    },
    "hack": {
        "score": 1.2
    }
}
```
-   `hacked.score` (`float`): 因被测出 Bug 而获得的分数
-   `hacked.count` (`int`): 被测出的 Bug 总数
-   `hacked.unfixed` (`int`): 未修复的 Bug 数
-   `hack.score` (`float`): 因成功提交他人 Bug 而获得的分数

#### D. `strong_test_details` 和 `uml_detailed_results` 结构
**类型结构**：`list[dict[Union...]]`
这两个字段都是列表，其中每个元素是一个**字典**，代表一个测试点的结果

**`strong_test_details`**
```json
{
    "message": "REAL_TIME_LIMIT_EXCEED",
    "score": 85.0
}
```
-   `message` (`str`): 测试点结果信息
    -   **分类值**: `'ACCEPTED'`, `'REAL_TIME_LIMIT_EXCEED'`, `'WRONG_ANSWER'`, `'CPU_TIME_LIMIT_EXCEED'` 等
-   `score` (`float`): 单个测试点得分

**`uml_detailed_results`**
```json
{
    "name": "类图强测1：类图正确性",
    "message": "ACCEPTED"
}
```
-   `name` (`str`): UML 检查项的名称
-   `message` (`str`): 检查结果**分类值**: `'ACCEPTED'` 或其他错误信息

#### E. `commits` 结构
**类型结构**：`list[dict[Union...]]`
`commits` 是一个列表，其中每个元素是一个**字典**，代表一次 Git 提交

```json
{
    "timestamp": "datetime.datetime(2025, 2, 25, 3, 42, tzinfo=...)",
    "message": "refactor: optimize performance"
}
```
-   `timestamp` (`datetime.datetime`): 提交时间戳，用于计算工作时长、昼夜节律等
-   `message` (`str`): 提交信息，用于分析关键词（如 `fix`, `refactor`, `v-`）以判断开发过程

#### F. `forum_activities` 结构
**类型结构**：`list[dict[Union...]]`
`forum_activities` 是一个列表，其中每个元素是一个**字典**，代表一次论坛互动

**通用字段:**
-   `type` (`str`): 互动的类型
    -   **分类值**: `'authored'` (自己发帖), `'commented'` (自己回帖)

**当 `type` 是 `'authored'` 时，使用的字段:**
```json
{
    "type": "authored",
    "title": "关于第三次作业的疑问",
    "category": "issue",
    "priority": "essential"
}
```
-   `title` (`str`): 帖子标题
-   `category` (`str`): 帖子分类
    -   **分类值**: `'issue'` (问题求助), `'free_discuss'` (自由讨论)
-   `priority` (`str`): 帖子优先级
    -   **分类值**: `'top'` (置顶), `'essential'` (精华), `'normal'` (普通)

**当 `type` 是 `'commented'` 时，使用的字段:**
```json
{
    "type": "commented",
    "post_title": "【问题反馈】指导书问题反馈专用帖",
    "post_author": "课程助教",
    "post_category": "issue",
    "post_priority": "top"
}
```
-   `post_title` (`str`): 所回复帖子的标题
-   `post_author` (`str`): 所回复帖子的作者名，用于区分是回复助教还是回复同学
-   `post_category` (`str`): 所回复帖子的分类，用途同上
-   `post_priority` (`str`): 所回复帖子的优先级，用途同上