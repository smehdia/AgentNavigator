import argparse

from dynaconf import Dynaconf

import os

from Agents.factory import build_agent
from Driver.factory import build_driver
from Debugger import Debugger

dbg = Debugger(palette="soft", indent_size=2, width=90)


for var in ["http_proxy", "https_proxy", "ftp_proxy", "socks_proxy", 
            "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "SOCKS_PROXY"]:
    os.environ.pop(var, None)


def ask_ok(prompt):
    return input(f"{prompt} (y/n): ").strip().lower() == "y"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    configs = Dynaconf(settings_files=[args.config], merge_enabled=True).default
    agent = build_agent(
        model_name=configs.agent.model_name,
        url=configs.agent.url,
        agent_settings=configs.agent.settings,
    )

    agent_settings = dict(configs.agent.settings)
    agent_settings["verbose_print"] = True  # required for agent debug panels
    agent = build_agent(
        model_name=configs.agent.model_name,
        url=configs.agent.url,
        agent_settings=agent_settings,
        debugger=dbg,
    )

    driver = build_driver(settings=configs.driver, agent=agent)


    expected = configs.driver.appPackage

    print("1) Opening app...")
    driver.run_application()
    driver.wait()
    print("OK" if ask_ok("App opened correctly?") else "FAIL")

    print("2) reset_to_start_page (close, reopen, agent reset_instruction)...")
    driver.reset_to_start_page()
    print("OK" if ask_ok("Reset landed on start screen?") else "FAIL")

    print("3) Foreground check loop (q to quit)")
    while True:
        nav = input("Go somewhere in the app, then press Enter (q to quit): ").strip().lower()
        if nav == "q":
            break
        fg = driver.get_foreground_package()
        match = fg == expected
        print(f"  foreground={fg!r}  expected={expected!r}  {'OK' if match else 'MISMATCH'}")
