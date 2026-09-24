import asyncio
import json
from dotenv import load_dotenv
load_dotenv()
from src.query.query_flow import build_tdd
from src.query.render import render_text

async def main():
    issue = "Payment authorization is failing for orders that have multiple shipping addresses"
    tdd = await build_tdd(issue, top_k=2, hops=1, debug_dump_prompts=True)

    text = render_text(tdd)
    print(text)

    with open("technical_design.txt", "w") as f:
        f.write(text)
    with open("technical_design.json", "w") as f:
        json.dump(tdd.model_dump(), f, indent=2)

    print("\nSaved to technical_design.txt and technical_design.json")

asyncio.run(main())
