import argparse
import asyncio
import pprint
import re
import time
import playwright
from playwright.async_api import async_playwright, Response
import playwright.async_api

# 定义目标域名
TARGET_DOMAIN = "api.oo.buaa.edu.cn/homework"
COURSE = "api.oo.buaa.edu.cn/course"
POSTS = r"^\w+://api\.oo\.buaa\.edu\.cn/homework/\d+/posts.*?$"
POST = "api.oo.buaa.edu.cn/post"
BASE_URL = "http://oo.buaa.edu.cn"

# 创建一个列表来存储捕获到的数据
captured_responses = []
courses = []
post_pages = []
posts = []

def _append(lists: list, element:any):
    has = False
    for e in lists:
        if e['url'] == element['url']:
            has = True
            break
    
    if not has:
        lists.append(element)

# 2. 定义一个异步的事件处理函数
async def handle_response(response: Response):
    """这个函数会在每次页面收到响应时被调用"""
    # 3. 过滤出目标域名的响应
    if TARGET_DOMAIN in response.url:
        print(f"[CAPTURED] 捕获到目标响应: {response.url}")
        print(f"  - 状态码: {response.status}")
        
        # 4. 尝试将响应体解析为 JSON
        #    使用 try-except 是个好习惯，因为响应体不一定总是合法的 JSON
        try:
            json_body = await response.json()
            print(f"  - 响应体 (JSON): {json_body}")
            
            # 将有用的信息存入列表
            _append(captured_responses, {
                "url": response.url,
                "status": response.status,
                "body": json_body,
            })
        except Exception as e:
            # 如果解析JSON失败，则作为文本读取
            print(f"  - 无法解析为 JSON，尝试读取文本...")
            text_body = await response.text()
            # 只打印前100个字符以避免刷屏
            print(f"  - 响应体 (Text): {text_body[:100]}...")

async def handle_course(response: Response):
    if COURSE in response.url:
        try:
            json_body = await response.json()
            courses.append({
                "url": response.url,
                "status": response.status,
                "body": json_body,
            })
        except Exception as e:
            print(f"[ERROR] {e}")
            # 如果解析JSON失败，则作为文本读取
            print(f"  - 无法解析为 JSON，尝试读取文本...")

async def handle_post_pages(response: Response):
    if re.match(POSTS, response.url):
        try:
            json_body = await response.json()
            post_pages.append({
                "url": response.url,
                "status": response.status,
                "body": json_body,
            })
        except Exception as e:
            print(f"[ERROR] {e}")
            # 如果解析JSON失败，则作为文本读取
            print(f"  - 无法解析为 JSON，尝试读取文本...")
        
async def handle_posts(response: Response):
    if POST in response.url:
        try:
            json_body = await response.json()
            # 去除不必要的内容
            json_body["data"]["post"]["content"] = json_body["data"]["post"]["content"][:100]
            posts.append({
                "url": response.url,
                "status": response.status,
                "body": json_body,
            })
        except Exception as e:
            print(f"[ERROR] {e}")
            # 如果解析JSON失败，则作为文本读取
            print(f"  - 无法解析为 JSON，尝试读取文本...")


async def load_page(context:playwright.async_api.BrowserContext, handler=handle_response):
    page = await context.new_page()
    # print(f"[INFO] 设置监听器，目标域名: {TARGET_DOMAIN}")
    page.on("response", handler)
    return page

async def login(page:playwright.async_api.Page, usr, pwd):
    print(f"INFO: Attempting login for user: {usr}")
    await page.goto("http://oo.buaa.edu.cn")
    await page.locator(".topmost a").click()
    await page.wait_for_selector("iframe#loginIframe", timeout=10000)
    iframe = page.frame_locator("iframe#loginIframe")
    await iframe.locator("div.content-con-box:nth-of-type(1) div.item:nth-of-type(1) input").fill(usr)
    await page.wait_for_timeout(1000)
    await iframe.locator("div.content-con-box:nth-of-type(1) div.item:nth-of-type(3) input").fill(pwd)
    await page.wait_for_timeout(1000)
    await iframe.locator("div.content-con-box:nth-of-type(1) div.item:nth-of-type(7) input").click()
    await page.wait_for_timeout(1000)
    print("INFO: Login form submitted.") # 添加日志

async def get_page(page:playwright.async_api.Page, url):
    await page.goto(url)
    await page.wait_for_timeout(5000)
    await page.close()

async def get_all_posts(context:playwright.async_api.BrowserContext, course_id:int):
    pages = f"{BASE_URL}/assignment/{course_id}/discussions"
    page = await load_page(context, handle_post_pages)
    await page.goto(pages)
    await page.wait_for_timeout(1000)
    try:
        pprint.pprint(post_pages)
        index = 1
        max_pages = post_pages[0]["body"]["data"]["total_page"]
        ids = []
        while index <= max_pages:
            total_cnt = post_pages[index - 1]["body"]["data"]["total_count"]
            ids.extend([post_pages[index - 1]["body"]["data"]["posts"][i]["id"] for i in range(total_cnt)])
            index += 1
        print(f"[INFO] ids is {ids}")
        await page.close()
        post = f"{BASE_URL}/assignment/{course_id}/discussion"
        tasks = []
        for _id in ids:
            url = f"{post}/{_id}"
            tasks.append(get_page(await load_page(context, handle_posts), url))
        return tasks

    except Exception as e:
        print(f"[ERROR] error when get posts pages{e}")

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
    pprint.pprint(courses)
    if courses:
        try:
            all_courses = courses[0]['body']['data']['courses']
            course_id = 0
            for course in all_courses:
                if re.match(r"^\d+面向对象设计与构造$", course['name']):
                    course_id = course['id']
            print(f"[INFO] course_id is {course_id}")
            courses.clear()
            await page.goto(f"{BASE_URL}/course/{course_id}")
            await page.wait_for_timeout(2000)
            if courses:
                all_courses = [course for course in courses if str(course_id) in course['url']][0]['body']['data']['homeworks']
                ids = [the_id['id'] for the_id in all_courses]
            print("[INFO] ids is ",end="")
            pprint.pprint(ids)
            return ids
            
        except Exception as e:
            print(f"[ERROR] error when get courses {e}")


async def main(_id, pwd):
    async with async_playwright() as p:
        # 1. 启动浏览器
        browser = await p.chromium.launch(
            headless=False  # 设置为 False 可以看到浏览器界面，方便调试
        )
        context = await browser.new_context()
        page = await context.new_page()

        # 6. 导航到触发 API 请求的页面
        #    注意：这里我用一个示例网站，你需要换成真正会调用 buaa API 的那个页面。
        #    例如，可能是某个课程平台或登录页面。
        try:
            print("[INFO] 导航到目标页面...")
            await login(page, _id, pwd)
            await page.close()
            # 7. 等待一段时间，确保所有异步请求都已完成
            course_ids = await get_courses(context)
            # await p.stop()
            tasks = []
            index = 1
            for course_id in course_ids:
                if index % 4 != 0:
                    tasks.append(get_all_pages(context, course_id))
                index += 1
            await asyncio.gather(*tasks)
            print("[INFO] 页面加载完成，等待额外1秒以捕获更多动态请求...")
            await page.wait_for_timeout(1000)

        except Exception as e:
            print(f"[ERROR] 页面导航或操作失败: {e}")

        # 8. 关闭浏览器
        print("[INFO] 测试完成，关闭浏览器...")
        await browser.close()

        # (可选) 打印所有收集到的数据
        print("\n--- 所有捕获到的响应摘要 ---")
        import json
        captured_responses.extend(posts)
        print(json.dumps(captured_responses, indent=2, ensure_ascii=False))
        json.dump(captured_responses, open("tmp.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    # 创建一个参数解析器
    parser = argparse.ArgumentParser(description="通过 Playwright 自动捕获北航OO课程网站的API数据。")
    
    # 添加必要的命令行参数：学号和密码
    parser.add_argument("student_id", help="用于登录的学号 (例如: 23371265)")
    parser.add_argument("password", help="对应的统一认证密码 (建议使用引号包裹)")
    
    # 解析命令行传入的参数
    args = parser.parse_args()
    
    # 运行主异步函数，并将解析到的参数传递进去
    asyncio.run(main(args.student_id, args.password))