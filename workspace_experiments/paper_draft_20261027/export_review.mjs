import {chromium} from "../chip_quality_20261017/manual_review/node_modules/playwright/index.mjs";
import {fileURLToPath, pathToFileURL} from "node:url";
import path from "node:path";
import fs from "node:fs/promises";
import assert from "node:assert/strict";

const root=path.dirname(fileURLToPath(import.meta.url));
const checks=path.join(root,"preview_checks");
await fs.mkdir(checks,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:"C:/Users/25791/AppData/Local/ms-playwright/chromium-1228/chrome-win64/chrome.exe"});
try{
  const page=await browser.newPage({viewport:{width:1440,height:1050}});
  await page.goto(pathToFileURL(path.join(root,"index.html")).href);
  await page.locator("figure img").evaluate(async img=>{await img.decode();});
  const images=await page.locator("img").evaluateAll(nodes=>nodes.map(img=>({src:img.getAttribute("src"),width:img.naturalWidth,height:img.naturalHeight,complete:img.complete})));
  assert(images.length>=1&&images.every(img=>img.complete&&img.width>500&&img.height>500));
  const text=await page.locator("body").innerText();
  assert(!/DATASET_REFS|POLAR_REFS|PERTURB_REFS/.test(text));
  assert(text.includes("1. Introduction")&&text.includes("Related Work"));
  const brokenCitations=await page.locator('a.citation').evaluateAll(nodes=>nodes.filter(a=>!document.querySelector(a.getAttribute("href"))).map(a=>a.textContent));
  assert.equal(brokenCitations.length,0,JSON.stringify(brokenCitations));
  await page.screenshot({path:path.join(checks,"desktop.png")});
  await page.locator("figure").screenshot({path:path.join(checks,"figure_caption.png")});
  // Float the full-width plate after the introduction in the paginated review copy.
  await page.evaluate(()=>{const figure=document.querySelector("figure");const marker=document.createElement("span");marker.id="figure-original-position";figure.before(marker);const heading=[...document.querySelectorAll("article h1")].find(h=>h.textContent.startsWith("2."));heading.before(figure);});
  await page.pdf({path:path.join(root,"manuscript_review.pdf"),preferCSSPageSize:true,printBackground:true});
  await page.evaluate(()=>{const marker=document.getElementById("figure-original-position");marker.replaceWith(document.querySelector("figure"));window.scrollTo(0,0);});
  await page.setViewportSize({width:390,height:844});
  const geometry=await page.evaluate(()=>({viewport:innerWidth,width:document.documentElement.scrollWidth}));
  assert(geometry.width<=geometry.viewport,JSON.stringify(geometry));
  await page.screenshot({path:path.join(checks,"mobile.png")});
  const chinese=await browser.newPage({viewport:{width:1440,height:1050}});
  await chinese.goto(pathToFileURL(path.join(root,"introduction_zh.html")).href);
  await chinese.locator("figure img").evaluate(async img=>{await img.decode();});
  assert(!(await chinese.locator("body").innerText()).includes("DATASET_REFS"));
  await chinese.screenshot({path:path.join(checks,"chinese.png")});
  const result={images,geometry,brokenCitations,checks:["local figure loaded","citation anchors resolved","no placeholder references","desktop and mobile layout","PDF exported","Chinese reading copy loaded"]};
  await fs.writeFile(path.join(checks,"receipt.json"),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
