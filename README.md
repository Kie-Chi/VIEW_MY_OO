使用须知

- 本项目用于爬取并分析北航OO正课即（"XX面向对象设计与构造"），报告内容取决于学号，不取决于年份等其他因素。如果因服务器前后端变化而导致无法使用，请耐心等待更新
- 只需要输入账号密码，即可获得独属于你的OO报告

爬取内容涵盖
- 强测数据
- 互测数据
- bug修复数据
- 讨论区数据
- gitlab的commit历史数据

报告内容涵盖
- 强测、互测、bug修复趋势图
- 四单元能力雷达图
- 个人亮点（成就）分析
- 学期表现分析（强测、互测、bug修复、开发风格、提交行为分析）
- 单元轨迹
- 互测策略分析
- (*可有)社区（讨论区）行为与互测房间生态分析
- 历次作业分析
- 成就墙

前置条件
- python > 3.10 (自用Python版本3.12.9，不过10以上问题不大)
- requirements.txt

快速使用
- 在`gift.py`同目录下新建`config.yml`，内容可参照`example/config.yml`
- 
    ```py
        pip install -r requirements.txt
        python gift.py
    ```

新增`mode`
- 在`tools`新建`newmode`,并新增函数`captrue.py, preprocess.py`用于处理数据，最终数据格式满足`interface.md`即可

开发时间仓促，如有bug，请提供.cache/tmp.json(cleanup: False)以及报错信息


