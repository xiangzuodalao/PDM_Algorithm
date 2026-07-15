from fastapi import FastAPI, Header
from fastapi.openapi.docs import get_swagger_ui_html
import uvicorn
from fastapi.staticfiles import StaticFiles

# app = FastAPI()

app = FastAPI(docs_url = None)
app.mount("/statics", StaticFiles(directory="statics"), name="statics")
@app.get("/docs",include_in_schema = False)
async def custom_docs():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url = '/statics/swagger-ui/swagger-ui-bundle.js',
        swagger_css_url = '/statics/swagger-ui/swagger-ui.css'
    )


@app.get("/hello")
async def say_hello(user_agent: str | None = Header(default=None)):
    return {
        "message": "Hello from User Service!",
        "received_user_agent": user_agent
    }

if __name__ == "__main__":
    
    # 启动在 8001 端口，对应 ROUTE_MAP 中的 user 服务
    uvicorn.run(app, host="0.0.0.0", port=8001)