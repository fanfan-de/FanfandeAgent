from mcp.server.fastmcp import FastMCP
import os
import pandas as pd
# Create an MCP server
mcp = FastMCP("Demo", json_response=True)


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


# Add a prompt
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }

    return f"{styles.get(style, styles['friendly'])} for someone named {name}."

@mcp.tool()
def get_excel_summary(file_path: str) -> dict:
    """
    获取一个 Excel 文件的缩略信息。
    """
    summary = {
        "file_name": os.path.basename(file_path),
        "file_size_bytes": os.path.getsize(file_path),
        "last_modified_timestamp": os.path.getmtime(file_path),
        "sheets": {}
    }

    try:
        # 获取所有 sheet 的名称
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        for sheet_name in sheet_names:
            sheet_info = {
                "name": sheet_name,
                "rows": None,
                "columns": None,
                "column_names": [],
                "data_types": {},
                "head_sample": None,
                "summary_statistics": None
            }

            try:
                # 尝试只读取部分数据进行概览，特别是对于大文件
                # 默认读取前N行，或者只读取少量列，以节省内存和时间
                df_sample = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5) # 只读前5行

                sheet_info["rows"], sheet_info["columns"] = df_sample.shape[0], df_sample.shape[1]
                
                # 如果文件很大，这里的 df_sample.shape[0] 可能不准确，只代表读入的行数
                # 更准确的行数需要读取整个文件或利用 openpyxl
                
                sheet_info["column_names"] = df_sample.columns.tolist()
                
                # 数据类型推断
                sheet_info["data_types"] = {col: str(dtype) for col, dtype in df_sample.dtypes.items()}
                
                # 前几行数据样本（转换为Markdown格式或JSON）
                sheet_info["head_sample"] = df_sample.head(3).to_markdown(index=False) # 前3行，Markdown格式

                # 仅对数值列生成统计摘要
                numeric_cols = df_sample.select_dtypes(include=['number'])
                if not numeric_cols.empty:
                    sheet_info["summary_statistics"] = numeric_cols.describe().to_markdown()

            except Exception as e:
                sheet_info["error"] = f"Error processing sheet '{sheet_name}': {str(e)}"
            
            summary["sheets"][sheet_name] = sheet_info

    except Exception as e:
        summary["error"] = f"Error reading Excel file: {str(e)}"

    return summary

if __name__ == "__main__":
    mcp.run()