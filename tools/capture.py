import argparse
import asyncio
import pprint
import re
import sys  # 引入 sys 模块
import playwright
from playwright.async_api import async_playwright, Response
import playwright.async_api

# homework map
hw_map = {
    "一": 1, 
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
    "十三": 13,
    "十四": 14,
    "十五": 15,
    "十六": 16
}

# 定义目标域名
TARGET_DOMAIN = "api.oo.buaa.edu.cn/homework"
COURSE = "api.oo.buaa.edu.cn/course"
POSTS = r"^\w+://api\.oo\.buaa\.edu\.cn/homework/\d+/posts.*?$"
POST = "api.oo.buaa.edu.cn/post"
GITLAB = r"^\w+://gitlab\.oo\.buaa\.edu\.cn/groups/oo_homework_\d+/-/.*?$"
BASE_URL = "http://oo.buaa.edu.cn"

# 创建一个列表来存储捕获到的数据
captured_responses = []
courses = []
post_pages = []
posts = []
homeworks = []

def exit_with_error(message: str):
    """打印一条致命错误信息到 stderr 并以状态码 1 退出脚本。"""
    print(f"\n[CRITICAL ERROR] {message}", file=sys.stderr)
    print("[INFO] Script terminated due to a critical error.", file=sys.stderr)
    sys.exit(1)

def _append(lists: list, element:any):
    for i, existing_element in enumerate(lists):
        if existing_element['url'] == element['url']:
            if existing_element['status'] != 200:
                lists[i] = element
            return
    lists.append(element)

async def handle_response(response: Response):
    """这个函数会在每次页面收到响应时被调用"""
    if TARGET_DOMAIN in response.url:
        print(f"[CAPTURED] 捕获到目标响应: {response.url}")
        print(f"  - 状态码: {response.status}")

        try:
            json_body = await response.json()
            _append(captured_responses, {
                "url": response.url,
                "status": response.status,
                "body": json_body,
            })
        except Exception:
            # 对于单个请求的失败，我们只记录错误，不退出
            print(f"  - [WARNING] 无法将响应解析为 JSON: {response.url}")
            try:
                text_body = await response.text()
                print(f"  - 响应体 (Text): {text_body[:100]}...")
            except Exception as e:
                print("  - [WARNING] 重定向问题（可忽略）或是其他问题")
                print(      f"{e}")

async def handle_course(response: Response):
    if COURSE in response.url:
        try:
            json_body = await response.json()
            _append(courses, {
                "url": response.url,
                "status": response.status,
                "body": json_body,
            })
        except Exception as e:
            print(f"[WARNING] 处理课程响应时出错: {e}")

async def handle_post_pages(response: Response):
    if re.match(POSTS, response.url):
        try:
            json_body = await response.json()
            _append(post_pages, {
                "url": response.url,
                "status": response.status,
                "body": json_body,
            })
        except Exception as e:
            print(f"[WARNING] 处理帖子分页响应时出错: {e}")

async def handle_posts(response: Response):
    if POST in response.url:
        try:
            json_body = await response.json()
            # 去除不必要的内容
            if json_body.get("data", {}).get("post", {}).get("content"):
                 json_body["data"]["post"]["content"] = json_body["data"]["post"]["content"][:30] + "..."
            _append(posts, {
                "url": response.url,
                "status": response.status,
                "body": json_body,
            })
        except Exception as e:
            print(f"[WARNING] 处理帖子内容响应时出错: {e}")

async def handle_homeworks(response: Response):
    if re.match(GITLAB, response.url):
        try:
            json_body = await response.json()
            _append(homeworks, {
                "url": response.url,
                "status": response.status,
                "body": json_body,
            })
        except Exception as e:
            print(f"[WARNING] 处理GitLab作业响应时出错: {e}")


async def load_page(context:playwright.async_api.BrowserContext, handler=handle_response):
    page = await context.new_page()
    if handler != None:
        page.on("response", handler)
    return page

async def login_course(page:playwright.async_api.Page, usr, pwd):
    print(f"INFO: Attempting to log into OO course website for user: {usr}")
    await page.goto("http://oo.buaa.edu.cn")
    await page.locator(".topmost a").click()
    await page.wait_for_selector("iframe#loginIframe", timeout=10000)
    iframe = page.frame_locator("iframe#loginIframe")
    await iframe.locator("div.content-con-box:nth-of-type(1) div.item:nth-of-type(1) input").fill(usr)
    await iframe.locator("div.content-con-box:nth-of-type(1) div.item:nth-of-type(3) input").fill(pwd)
    await iframe.locator("div.content-con-box:nth-of-type(1) div.item:nth-of-type(7) input").click()
    await page.wait_for_url(f"{BASE_URL}/home", timeout=10000)
    print("INFO: OO course website login successful.")

async def login_gitlab(page:playwright.async_api.Page, usr, pwd):
    print(f"INFO: Attempting to log into GitLab for user: {usr}")
    await page.goto("http://gitlab.oo.buaa.edu.cn")
    try:
        await page.locator(".gl-button-text").click(timeout=5000)
        await page.wait_for_selector("iframe#loginIframe", timeout=10000)
        iframe = page.frame_locator("iframe#loginIframe")
        await iframe.locator("div.content-con-box:nth-of-type(1) div.item:nth-of-type(1) input").fill(usr)
        await iframe.locator("div.content-con-box:nth-of-type(1) div.item:nth-of-type(3) input").fill(pwd)
        await iframe.locator("div.content-con-box:nth-of-type(1) div.item:nth-of-type(7) input").click()
        await page.wait_for_selector("header.navbar-gitlab", timeout=15000)
        print("INFO: GitLab login successful.")
    except playwright.async_api.TimeoutError:
        print("[INFO] Already logged into GitLab or no login button found. Proceeding...")
    except Exception as e:
        # 其他未预料的错误是致命的
        exit_with_error(f"An unexpected error occurred during GitLab login: {e}")

async def get_page(page:playwright.async_api.Page, url):
    try:
        await page.goto(url)
        await page.wait_for_load_state("networkidle", timeout=10000)
        await asyncio.wait_for(page.close(), timeout=5)
    except (playwright.async_api.TimeoutError, asyncio.TimeoutError) as e:
        print("[WARNING] Network error, please wait for a while and try again")

async def get_all_posts(context:playwright.async_api.BrowserContext, course_id:int):
    pages_url = f"{BASE_URL}/assignment/{course_id}/discussions"
    page = await load_page(context, handle_post_pages)
    await page.goto(pages_url)
    await page.wait_for_timeout(2000) # 等待API响应
    
    the_pages = [p for p in post_pages if str(course_id) in p["url"] and p.get("body", {}).get("data")]
    if not the_pages:
        print(f"[WARNING] No post page data found for course_id {course_id}. Skipping post retrieval.")
        await page.close()
        return []

    try:
        max_pages = the_pages[0]["body"]["data"]["total_page"]
        ids = []
        for p_data in the_pages:
            if "posts" in p_data["body"]["data"]:
                ids.extend([post_item["id"] for post_item in p_data["body"]["data"]["posts"]])
        
        ids = list(set(ids))
        print(f"[INFO] Found {len(ids)} post IDs for course {course_id}.")

        await page.close()

        tasks = []
        post_base_url = f"{BASE_URL}/assignment/{course_id}/discussion"
        for _id in ids:
            url = f"{post_base_url}/{_id}"
            new_page = await load_page(context, handle_posts)
            tasks.append(get_page(new_page, url))
        return tasks

    except (KeyError, IndexError) as e:
        print(f"[WARNING] Could not parse post pages for course {course_id}: {e}. Skipping...")
        await page.close()
        return []

async def get_all_pages(context:playwright.async_api.BrowserContext, course_id:int):
    assessment = f"{BASE_URL}/assignment/{course_id}/assessment"
    mutual = f"{BASE_URL}/assignment/{course_id}/mutual"
    bugfix = f"{BASE_URL}/assignment/{course_id}/bugFixes"
    task_a = get_page(await load_page(context), assessment)
    task_b = get_page(await load_page(context), mutual)
    task_c = get_page(await load_page(context), bugfix)    
    task_posts = await get_all_posts(context, course_id)
    await asyncio.gather(
        task_a,
        task_b,
        task_c,
        *task_posts
    )

async def get_courses(context:playwright.async_api.BrowserContext):
    page = await load_page(context, handle_course)
    await page.goto(f"{BASE_URL}/courses")
    await page.wait_for_timeout(2000)

    if not courses:
        exit_with_error("Failed to capture any course data. Cannot proceed.")

    try:
        all_courses = courses[0]['body']['data']['courses']
        course_id = 0
        for course in all_courses:
            if re.match(r"^\d+面向对象设计与构造$", course['name']) and course['role'] == '学生':
                course_id = course['id']
                break
        if not course_id:
             exit_with_error("Could not find the '面向对象设计与构造' student course.")
        courses.clear()
        await page.goto(f"{BASE_URL}/course/{course_id}")
        await page.wait_for_url(f"{BASE_URL}/course/{course_id}", timeout=5000)
        await page.wait_for_timeout(2000)
        if not courses:
            exit_with_error(f"Failed to capture homework data for course ID {course_id}.")

        homework_data = [c for c in courses if str(course_id) in c['url'] and c.get("body", {}).get("data", {}).get("homeworks")]
        if not homework_data:
            exit_with_error(f"No valid homework list found in API response for course ID {course_id}.")

        all_homeworks = homework_data[0]['body']['data']['homeworks']
        ids = [hw['id'] for hw in all_homeworks]
        id_map = {hw['id']: hw_map[re.findall(r"第(\w+)次作业", hw['name'])[0]] for hw in all_homeworks}
        print("[INFO] Found homework IDs: ", end="")
        pprint.pprint(ids)
        await page.close()
        return ids, id_map

    except (KeyError, IndexError, TypeError) as e:
        exit_with_error(f"Failed to parse course or homework data. Error: {e}")

commits = []
lock = asyncio.Lock()

async def get_all_commits(context:playwright.async_api.BrowserContext, hw_tpl: tuple):
    try:
        hw = hw_tpl[0]
        url = f"{hw}commits"
        page = await load_page(context, None)
        await page.goto(url)
        await page.wait_for_load_state("networkidle")
        _commits_elements = await page.locator("li > div.commit-detail .commit-content > a").all()
        _times_elements = await page.locator("li > div.commit-detail .commit-content time").all()
        _commits = [await el.inner_text() for el in _commits_elements]
        _times = [await el.get_attribute("title") for el in _times_elements]

        if len(_times) != len(_commits):
             print(f"[WARNING] Mismatch in commit messages and timestamps for {hw}. Skipping this homework.")
             return
        
        _all = {_times[i]: _commits[i] for i in range(len(_commits))}

        def _append_commit(lists: list, element: any):
            for e in lists:
                if e['hw'] == element['hw']:
                    return
            lists.append(element)

        async with lock:
            _append_commit(commits, {"hw": hw_tpl[1], "commits": _all})
        await page.close()

    except Exception as e:
        print(f"[WARNING] Failed to get commits from GitLab for {hw}. Error: {e}")


async def get_homeworks_urls(id_map: dict):
    ultimate_selfs = [
        r for r in captured_responses 
        if "ultimate_test/submit" in r["url"] and r["status"] == 200 and r.get("body", {}).get("data")
    ]
    
    # 作业数量可能不是固定的12，进行灵活处理
    if len(ultimate_selfs) != 12:
        exit_with_error(f"gitlab url has {len(ultimate_selfs)} but we only have 12 homework")
    
    print(f"[INFO] Found {len(ultimate_selfs)} homework entries.")
    urls = [(resp["body"]["data"]["gitlab_url"], id_map[int(re.findall(r".*?/(\d+)/.*?", resp["url"])[0])]) for resp in ultimate_selfs if resp["body"]["data"].get("gitlab_url")]
    if not urls:
        exit_with_error("Could not extract any GitLab URLs from the API responses.")
        
    return urls


async def main(_id, pwd, mode):
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=not mode
        )
        context = await browser.new_context()
        page = await context.new_page()
        id_map = {}
        try:
            print("[INFO] Starting scraping process for OO course website...")
            await login_course(page, _id, pwd)
            await page.close()

            course_ids, id_map = await get_courses(context)
            pprint.pprint(id_map)
            # 按批次执行，避免一次性打开太多页面
            split = 3
            tasks = [get_all_pages(context, course_id) for course_id in course_ids]
            
            # 过滤掉第4, 8, 12...次作业（通常是训练）
            filtered_tasks = [task for i, task in enumerate(tasks) if (i + 1) % 4 != 0]

            split_tasks = [filtered_tasks[i:i + split] for i in range(0, len(filtered_tasks), split)]
            
            for i, batch in enumerate(split_tasks):
                print(f"[INFO] --- Running batch {i+1}/{len(split_tasks)} of page scraping ---")
                await asyncio.gather(*batch)
                
        except Exception as e:
            if isinstance(e, playwright.async_api.Error):
                 exit_with_error(f"A Playwright operation failed: {e}")
            else:
                 exit_with_error(f"An unexpected error occurred during course data scraping: {e}")
        await asyncio.sleep(3)

        try:
            print("[INFO] Starting scraping process for GitLab...")
            page = await context.new_page()
            await login_gitlab(page, _id, pwd)
            await page.close()

            hw_urls = await get_homeworks_urls(id_map)
            capture_regex = r"(.*?/-/).*?"
            base_hw_urls = list(set([(re.findall(capture_regex, hw_url[0])[0], hw_url[1]) for hw_url in hw_urls if re.findall(capture_regex, hw_url[0])]))
            pprint.pprint(base_hw_urls)
            print(f"[INFO] Found {len(base_hw_urls)} unique GitLab homework repositories.")
            # pprint.pprint(base_hw_urls)
            
            tasks = [get_all_commits(context, hw) for hw in base_hw_urls]
            await asyncio.gather(*tasks)

        except Exception as e:
            exit_with_error(f"An unexpected error occurred during GitLab scraping: {e}")

        print("[INFO] Scraping complete, closing browser...")
        await browser.close()

        print("\n--- Aggregating and saving results ---")
        captured_responses.extend(posts)
        captured_responses.extend(commits)
        
        output_file = "tmp.json"
        with open(output_file, "w", encoding="utf-8") as f:
            import json
            json.dump(captured_responses, f, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] All data has been saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture API data from BUAA OO course website.")
    parser.add_argument("student_id", help="Your student ID (e.g., 23371265)")
    parser.add_argument("password", help="Your SSO password (use quotes if it contains special characters)")
    parser.add_argument("debugmode", help="Use debug mode to detemine headless")
    args = parser.parse_args()
    
    asyncio.run(main(args.student_id, args.password, bool(args.debugmode)))