// Records the two halves of docs/assets/fieldtrial.gif (see docs/assets/README.md).
// Usage: fieldtrial demo --dir /tmp/demo-study --port 8798 --no-browser, then
//        node record_hero.mjs OUT_DIR   (needs the playwright package)
import { createRequire } from "module";
import fs from "fs";
const require = createRequire(process.env.NODE_TOOLS || import.meta.url);
const { chromium } = require("playwright");
const BASE = process.env.FIELDTRIAL_URL || "http://127.0.0.1:8798";
const OUT = process.argv[2];
const t0 = Date.now();
const marks = {};
const mark = (k) => { marks[k] = (Date.now() - t0) / 1000; };
const browser = await chromium.launch();
const phone = await browser.newContext({ viewport: { width: 390, height: 780 }, deviceScaleFactor: 1,
  recordVideo: { dir: OUT + "/phone", size: { width: 390, height: 780 } } });
const desk = await browser.newContext({ viewport: { width: 960, height: 780 }, deviceScaleFactor: 1,
  recordVideo: { dir: OUT + "/desk", size: { width: 960, height: 780 } } });
const page = await phone.newPage();
const mon = await desk.newPage();
await mon.goto(BASE + "/");
await page.goto(BASE + "/");
mark("start");
await page.waitForTimeout(1200);
await page.fill("input[name=operator]", "Ana");
await page.fill("input[name=rig]", "rig-1");
for (const box of await page.$$("input[type=checkbox][name^=check_]")) { await box.check(); await page.waitForTimeout(150); }
await page.waitForTimeout(400);
await page.click("button:has-text('Start session')");
await page.waitForURL(/\/sessions\//);
await mon.goto(page.url() + "/mirror");
await page.waitForTimeout(1200);
mark("session");
const stages = [4, 2, 4, 1, 4];  // keys: 4 = success here, others partial
for (const key of stages) {
  await page.keyboard.press("Space");
  await page.waitForSelector(".timer");
  await page.waitForTimeout(2200);
  await page.keyboard.press("Space");
  await page.waitForSelector("form.label-form");
  await page.waitForTimeout(700);
  await page.keyboard.press(String(key));
  await page.waitForTimeout(600);
  if (key !== 4) {
    await page.click("label:has(input[name=termination][value=timeout])");
    await page.waitForTimeout(400);
    const tag = await page.$("label:has(input[name=failure_tags])");
    if (tag) { await tag.click(); await page.waitForTimeout(400); }
  }
  await page.keyboard.press("Enter");
  await page.waitForSelector("form.label-form", { state: "detached" });
  await page.waitForTimeout(900);
}
mark("visible_end");
while (!(await page.isVisible("text=All scheduled trials are done"))) {
  if (await page.$("form.label-form")) { await page.keyboard.press("Enter"); await page.waitForSelector("form.label-form", { state: "detached" }); continue; }
  await page.keyboard.press("Space"); await page.waitForSelector(".timer");
  await page.keyboard.press("Space"); await page.waitForSelector("form.label-form");
}
await page.waitForTimeout(500);
mark("done");
await mon.goto(BASE + "/studies/demo-study/report");
await mon.waitForTimeout(1500);
mark("unblind_page");
await mon.check("input[name=confirm]");
await mon.waitForTimeout(600);
await mon.click("button:has-text('Unblind')");
await mon.waitForSelector("text=Success rate per arm");
await mon.goto(BASE + "/studies/demo-study/report.html");
await mon.waitForTimeout(1800);
mark("report");
for (let y = 0; y < 2400; y += 40) { await mon.evaluate((v) => window.scrollTo(0, v), y); await mon.waitForTimeout(60); }
await mon.waitForTimeout(1500);
mark("end");
await phone.close(); await desk.close(); await browser.close();
fs.writeFileSync(OUT + "/marks.json", JSON.stringify(marks));
console.log(JSON.stringify(marks));
