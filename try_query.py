import asyncio
import json
from dotenv import load_dotenv
load_dotenv()
from src.query.pdf_render import render_pdf
from src.query.query_flow import build_tdd
from src.query.render import render_text
from src.tools.query_tools import _publish_to_confluence

async def main():

    issue = (
        "There are two separate problems to fix. First, in the payment system, we don't currently "
        "have a way to specify a currency alongside a payment amount — we'd like to add a currency "
        "field to payment requests and validate that the given currency code is actually supported "
        "before processing the payment. Second, and unrelated to the first, bank transfers have two "
        "problems: withdrawing more money than an account's current balance is allowed to go through "
        "instead of being rejected, and transfers of zero or negative amounts are also allowed "
        "instead of being rejected."
    )

    tdd = await build_tdd(issue, top_k=5, hops=1, debug_dump_prompts=True)

    text = render_text(tdd)
    print(text)

    with open("technical_design.txt", "w") as f:
        f.write(text)
    with open("technical_design.json", "w") as f:
        json.dump(tdd.model_dump(), f, indent=2)
    with open("technical_design.pdf", "wb") as f:
        f.write(render_pdf(tdd))

    print("\nSaved to technical_design.txt, technical_design.json, and technical_design.pdf")

    confluence_url = _publish_to_confluence(tdd)
    if confluence_url:
        print(f"Published to Confluence: {confluence_url}")
    else:
        print("Confluence publish skipped (not configured, or it failed — see any [CONFLUENCE SKIP] line above).")

asyncio.run(main())
