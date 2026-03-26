# RAG 學習與實踐計畫

> **目標**：在 4 月初前掌握 RAG 核心概念、建立可評估的 RAG 系統
> 
> **預計投入**：2-3 週（15-20 小時）
> 
> **最終成果**：可跑的本地知識問答系統 + 評估報告

---

## 📚 教材選用與學習順序

### 推薦教材清單

| 優先級 | 教材名稱 | 來源 | 用途 | 預計時間 |
|--------|--------|------|------|---------|
| 1️⃣ **必讀** | RAG 基礎入門教程 | 中文綜合教程 | 完整知識框架 | 3-4 小時 |
| 2️⃣ **必讀** | LangChain RAG 官方教程 | https://python.langchain.com/docs/tutorials/rag/ | 動手實現 + 生態 | 3-4 小時 |
| 3️⃣ **必讀** | LlamaIndex 入門指南 | https://docs.llamaindex.ai/ | 資料索引 & 檢索策略 | 2-3 小時 |
| 4️⃣ 參考 | OpenAI Cookbook RAG | https://cookbook.openai.com/ | 實作細節 + 評估思路 | 2-3 小時 |
| 5️⃣ 參考 | Haystack RAG Pipeline | https://haystack.deepset.ai/tutorials | 工程化架構 | 2 小時 |
| 6️⃣ 參考 | Pinecone 向量資料庫 | https://www.pinecone.io/learn/ | 生產環境部署 | 1-2 小時 |

**學習策略**：
- ✅ 第 1-3 個教材按順序學（搭建基礎 + 代碼能力）
- ✅ 第 4-6 個教材作為問題時才查閱（不背誦，按需參考）

---

## 🎯 分階段學習計畫

### 第一階段：基礎概念 & 最小可跑版本（第 1-3 天）

目標：理解 RAG 架構，能跑通一個簡單 demo

#### 1.1 RAG 核心概念與系統架構深度解析

##### 📖 第一部分：什麼是 RAG？概念、問題背景與必要性

在開始 RAG 學習之前，必須深入理解「為什麼我們需要 RAG」這個根本問題。看似簡單的一句「Retrieval Augmented Generation」背後，實則反映了當今大型語言模型面臨的深層困境。

**定義與核心概念**

RAG（檢索增強生成）是一種混合型 AI 技術架構，其核心思想是：**不依賴模型預訓練知識，而是在生成過程中即時從外部知識庫檢索相關信息，與用戶查詢結合，輸入給 LLM 生成答案**。換句話說，它將「知識存儲」（知識庫）與「知識應用」（LLM）分開，通過動態檢索來彌補 LLM 的知識限制。

這個看似簡單的思路實際上解決了 LLM 時代的一個本質矛盾：**訓練時間 vs. 知識實時性**。傳統微調方案要求定期對模型重新訓練才能更新知識，而 RAG 只需要更新知識庫文件，大幅降低了維護成本。

**深度問題分析：為什麼大型語言模型需要補強？**

當前所有主流 LLM（GPT-4、Claude、Gemini 等）都存在以下四大固有限制，這些限制直接導致 RAG 的出現：

**1. 知識截止日期問題（Knowledge Cutoff）**

GPT-3.5 的訓練數據截止於 2021 年 4 月，GPT-4 約為 2024 年 4 月。這意味著：
- 發生於截止日期之後的任何事件，模型完全無法理解
- 2024 年底「某公司新產品發布」、「新政策出台」等，模型一無所知
- 對於不斷演進的技術領域（AI、編程框架、API 更新），模型知識快速老化
- 實例：用 GPT-3.5 詢問「2024 年 Python 3.12 有什麼新特性」，模型會拒絕或編造

**2. 幻覺問題（Hallucination）**

LLM 會以極高的自信度編造不存在的信息。這不是故意欺騙，而是模型一個內在的數學特性：
- 給定 Token 序列，模型基於統計概率預測下一個 Token
- 當遇到超出訓練分布的問題時，模型會「創造」回答
- 典型例子：
  ```
  Q: 「周傑倫的第 20 張專輯名叫什麼？」
  A (GPT-3.5): 「《未來夢想》，發行於 2023 年...」（純屬編造，周傑倫沒有這張專輯）
  ```
- 統計顯示，在開放域問題上，LLM 的幻覺率可高達 30-50%
- 幻覺對企業應用極具破壞性：法律文件引用錯誤、醫療建議錯誤、財務數據錯誤都可能導致嚴重後果

**3. 領域知識缺乏（Domain Knowledge Gap）**

通用 LLM 不可能掌握任何組織的私有知識。這包括：
- **企業內部文檔**：公司內部流程、產品文檔、代碼庫說明、決策歷史
- **個人知識**：個人管理系統、筆記、學習記錄、項目文檔
- **隱私信息**：客戶數據、合同、機密戰略規劃
- **新興領域**：新興學科、小眾行業、特定公司的學問

即使這些知識本身不算罕見，但涉及隱私時將其上傳給 OpenAI 或 Anthropic 就變成了風險問題。

**4. 成本與資源限制（Fine-tuning Cost）**

微調方案雖然理論上可行，但成本極高：
- 需要 2000-5000 個標註訓練樣本（數月標註工作）
- 需要 GPU 資源（A100 級別，$3000-5000/月租賃成本）
- 調試和優化週期長（1-2 周）
- 每次更新知識都要重新訓練（迭代慢）
- 微調後模型容易 catastrophic forgetting（遺忘其他知識）

成本計算：採用微調方案 = 標註成本 ¥30,000-50,000 + GPU 租賃 $500-1000 × 2 周 + 工程師工資 = 總成本 ¥100,000 以上。相比之下，RAG 成本不到微調的 1/10。

**RAG 如何解決這些問題**

每個 LLM 的基本限制都有對應的 RAG 解決方案：

| 問題 | LLM 局限 | RAG 解決方案 | 效果 |
|------|---------|-----------|------|
| 知識過期 | 訓練數據截止，無法更新 | 知識庫文件實時可更新，無需重訓 | ✅ 秒級更新，可擴展至 GB-TB 規模 |
| 幻覺率高 | 無法知道自己不知道什麼 | 檢索結果明確，填補知識空白，促進事實準確 | ✅ 幻覺率從 30-50% 降至 < 10% |
| 領域知識缺 | 跨越訓練分布的知識毫無辦法 | 將專有文檔存入知識庫，檢索時提供精確上下文 | ✅ 可適配任何領域，0 知識洩露 |
| 成本巨大 | 微調需數百萬成本 | 只需向量搜索 + API 調用，無需訓練 | ✅ 成本 1/10 或更低，部署時間 2-3 天 |

**加強讀物技巧：理解 RAG 為什麼有效**

RAG 之所以有效，從信息論角度看，是因為它大幅降低了 LLM 生成的**熵（uncertainty）**。在沒有上下文的情況下，LLM 面對一個開放問題時熵很高（可能的回答太多），而一旦提供了精確的檢索結果作為上下文，模型的選擇空間就被限制在現有文檔中，從而大幅提升準確性。

打個比喻：
- **不用 RAG**：「根據你的知識，宇航員登上火星的挑戰是什麼？」→ LLM 自由發揮，高概率編造
- **用 RAG**：「根據以下文章，宇航員登上火星的挑戰是什麼？[文章內容...]」→ LLM 只能從給定文章提取，準確性大幅提升

---

**RAG 與微調的全方位對比**

很多初學者會問：「既然微調也能增強 LLM，為什麼要用 RAG？」答案是：**微調和 RAG 各有所長，選擇應該基於使用場景**。

| 維度 | RAG | 微調 | 何時選擇 RAG | 何時選擇微調 |
|------|-----|------|----------|-----------|
| **部署時間** | 2-3 天 | 1-2 週（含準備） | ✅ 快速上線需求 | ❌ 可接受延遲 |
| **成本** | $10-50/月（API + 存儲） | $500-5000（GPU + 標註） | ✅ 成本受限 | ❌ 預算充足 |
| **更新頻率** | 即時（秒級） | 數天到數週 | ✅ 知識頻繁變化 | ❌ 知識相對穩定 |
| **知識容量** | 無上限（GB-TB） | 受模型大小 10%-50%限制 | ✅ 超大知識庫 | ❌ 知識量可控 |
| **準確性** | 很高（基於真實文檔） | 變數大（依賴數據質量） | ✅ 對準確性要求高 | ❌ 樂意接受不確定性 |
| **可解釋性** | 完美（引用原文檔） | 低（黑盒） | ✅ 需要可追溯性 | ❌ 可接受黑盒 |
| **知識遷移** | 自動對接新領域 | 需完全重新訓練 | ✅ 跨領域應用 | ❌ 單一領域深耕 |
| **隱私性** | 完全保護（私有知識庫） | 數據進入第三方服務器 | ✅ 涉及敏感信息 | ❌ 公開信息 |
| **維護難度** | 簡單（更新文件） | 複雜（重新標註+訓練） | ✅ 團隊規模小 | ❌ 有專門 ML 團隊 |

**進階技巧：混合方案**

實際上，最優的架構是 **RAG + 微調的混合方案**：
1. 用 RAG 處理 80% 的常見問題（快速、低成本）
2. 用微調處理 20% 的領域特定高價值問題（高精度）
3. 針對特定風格要求（寫詩、編代碼風格）使用微調
4. 對於需要多步推理的複雜問題，先用 RAG 檢索背景，再用微調模型生成

---

**應用場景速查表：何時選擇 RAG**

根據以下特徵快速確定是否適合 RAG：

| 場景 | 適合 RAG 嗎 | 理由 | 推薦配置 |
|------|---------|------|---------|
| 企業內部文檔問答系統 | ✅ 適合 | 知識庫穩定，需隱私，無需風格轉變 | FAISS + GPT-3.5 |
| 實時新聞 FAQ 系統 | ✅ 最適合 | 需頻繁更新、無需微調 | Pinecone + GPT-3.5 |
| 個人知識庫系統（筆記、Blog 搜索） | ✅ 適合 | 私有知識、量大、無風格需求 | Chroma + 本地模型 |
| 法律文件查詢系統 | ✅ 最適合 | 需高準確性、可追溯性、隱私 | RAG + 律師審閱 |
| 編程助手（代碼補全、API 查詢） | ✅ 適合 | API 文檔多、頻繁變化、需精確匹配 | BM25 + 向量混合搜索 |
| 創意寫作（詩歌、故事） | ❌ 不適合 | 需要風格轉變，微調更好 | 微調或 RAG + CoT |
| 代碼審查指導（風格糾正） | ❌ 需要微調 | 涉及風格和習慣改變 | 微調 |
| 複雜推理題（數學、邏輯） | ⚠️ 混合最好 | 背景信息用 RAG，推理用微調 | RAG 檢索 + 微調生成 |
| 客服對話助手 | ✅ 適合 | 回答一致性需求、FAQ 完整 | RAG + 多輪對話管理 |
| 醫療診斷助手 | ✅ 適合 | 需要最新臨床指南、可落責 | RAG + 醫生審閱 |

---

##### 🏗️ 第二部分：RAG 系統的三大核心構件全面解析

一個完整的 RAG 系統並非一個整體，而是由三個相對獨立又緊密配合的組件構成。理解這三個構件的角色、工作原理和常見問題，是掌握 RAG 的關鍵。

**構件 1：檢索器（Retriever）— RAG 系統的前哨**

*角色與使命*

檢索器的使命很清晰：**給定用戶查詢，從龐大的知識庫中找出最相關的1-10份文檔片段**。它是 RAG 系統的「眼睛」，決定了系統能否看見正確的信息。

*工作流程的詳細解析*

讓我們追蹤一個具體的查詢如何通過檢索器處理：

```
1️⃣ 用戶提問：「微調訓練中 weight_decay 參數設為多少最佳？」

2️⃣ 查詢向量化
   - 入力：中文自然語言查詢 (20 個字)
   - 嵌入模型：OpenAI text-embedding-3-small
   - 輸出：512 維向量（每個維度是 -1 到 1 的浮點數）
   - 含義：第一維可能代表「超參優化」概念，第三維代表「訓練穩定性」等
   - 計算時間：~100-200 ms

3️⃣ 相似度計算（向量空間搜索）
   - 方法：餘弦相似度 cos(θ) = (A·B)/(|A||B|)
   - 對比對象：知識庫中所有 5000 個文檔片段（也都已向量化）
   - 時間複雜度：O(n) = 5000 × 計算 ~50ms
   - 結果：5000 個相似度分數 (0 到 1 之間)

4️⃣ Top-K 檢索
   - K = 5（通常選擇）
   - 取出相似度最高的 5 個文檔片段
   - 相似度範圍：通常 0.75-0.95（相對很高）

5️⃣ 相關性過濾（可選，防止噪聲）
   - 若設置相似度閾值 > 0.6，低於此分數的結果被拋棄
   - 實例：如果 Top-5 中第 3 個文檔相似度只有 0.58，被篩掉了

6️⃣ 返回結果
   - 返回給下一階段的上下文：
   ```
   [
     (分數 0.92) "weight_decay 用於 L2 正則化，建議值為 1e-4 到 1e-2...",
     (分數 0.88) "在 Adam 優化器中，weight_decay 等效於...",
     (分數 0.84) "實驗顯示 weight_decay=1e-5 時過擬合最少...",
     ...
   ]
   ```
   - 這些文本被拼接成一個大段落，成為 LLM 的「背景知識」
```

*檢索質量的量化評估*

如何判斷檢索器靠不靠譜？用以下三個核心指標（都是你後續評估和優化時需要計算的）：

**Recall@K（召回率）**
- 定義：假設某個查詢的相關文檔有 M 份，Top-K 檢索中命中了 R 份，則 Recall@K = R/M
- 直白例子：
  ```
  查詢：「weight_decay 最佳實踐」
  知識庫中所有相關文檔：10 份（我們通過人工標註知道）
  Top-5 檢索結果：命中了 4 份相關文檔
  Recall@5 = 4/10 = 40%
  
  這表示系統漏掉了 60% 的相關信息，不夠好。
  目標通常是 Recall@5 ≥ 80%，即找到 8 份相關文檔中的至少 7 份。
  ```
- 為什麼重要？Recall 高意味著 LLM 能看見充分的背景信息，生成答案會更全面、更少幻覺。

**MRR (Mean Reciprocal Rank)**
- 定義：衡量最相關文檔排在第幾位。計算方式是 1/(首個相關文檔的排名)
- 例子：
  ```
  查詢：「weight_decay 最佳實踐」
  檢索結果排序：
    1. [相似度 0.95] 「重要：weight_decay 建議 1e-4...」← 這是相關文檔
    2. [相似度 0.92] 「Adam 優化器介紹」
    3. [相似度 0.85] 「L2 正則化原理」
  
  MRR = 1/1 = 1.0（最好的情況）
  
  反面例子：
    1. [相似度 0.98] 「優化器概述」← 無關
    2. [相似度 0.96] 「超參搜索工具」← 無關
    3. [相似度 0.92] 「weight_decay 建議 1e-4...」← 相關文檔在第 3 位
  
  MRR = 1/3 = 0.33（不夠好，相關文檔排得太後面）
  ```
- 為什麼重要？MRR 高意味著檢索結果按相關性排序得很好，LLM 會優先看見最相關的文檔。

**NDCG@K (Normalized Discounted Cumulative Gain)**
- 定義：綜合考慮相關文檔的排名位置，給排名靠前的相關文檔更高的權重
- 公式簡化理解：$NDCG@K = \frac{1}{IDCG} \sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_2(i+1)}$
  - $rel_i$ = 第 i 個結果的相關程度（0 = 無關，1 = 相關，2 = 非常相關）
  - 越往前的結果（i 越小），$\log_2(i+1)$ 越小，相關性被「折現」越少（權重越高）
- 實際應用不用手算，LangChain 或專業評測工具會自動計算

*常見檢索失敗模式與診斷方法*

| 失敗現象 | Root Cause | 診斷方法 | 解決方案 |
|---------|----------|---------|---------|
| 返回全是无關文檔 | chunk_size 太大（500+ 字），核心信息被淹沒 | 查看檢索到的文本，看有多少百分比是相關的 | 改小 chunk_size 至 512-1024 |
| 始終返回同一份文檔 | 向量庫有重複文檔或分塊方式帶入重複 | 檢查向量庫中是否有完全相同的向量 | 去重後重建向量庫 |
| 找不到明顯相關的文件 | Query 措辭與知識庫用語差異大（同義詞問題） | 用相同關鍵詞在知識庫搜索，確認是否存在 | 啟用 Query Rewriting 或使用混合搜索 |
| 相似度分數都特別低 | 嵌入模型不適配該領域 | 檢查平均相似度是否 < 0.5 | 嘗試領域特定的嵌入模型或微調 |

---

**構件 2：分塊器（Chunker）— 知識庫的「大廚」**

*為什麼分塊是必要的？*

如果說檢索器是 RAG 的「眼睛」，分塊器就是「刀工師傅」。沒有正確的分塊，再好的檢索器也發揮不了作用。

想像一個極端場景：

```
❌ 壞做法：把一份 100KB 的技術文檔當作整體向量化
  - OpenAI embedding 模型限制 8192 tokens，但該文檔有 30,000 tokens
  - 被迫截斷，丟失 70% 內容
  - 向量無法充分表達文檔語義

✅ 好做法：把文檔分成 30 個 1000-token 的片段，分別向量化
  - 每個 chunk 都完整向量化，保留所有語義信息
  - 用戶查詢能準確匹配某個片段，而不是全文
  - 記憶體使用更高效
```

*分塊帶來的直接影響*

分塊決定了：
1. **檢索精度**：塊太大 → 核心信息被雜訊埋沒；塊太小 → 上下文丟失
2. **成本**：一個知識庫被分成 100 個 chunks vs 1000 個 chunks，成本差 10 倍
3. **延遲**：塊數多 → 檢索速度慢，搜索 10,000 個 chunks 比搜索 1000 個慢 10 倍
4. **LLM 上下文窗口利用率**：chunk 太大，LLM 提示詞被上下文佔滿；chunk 太小，LLM 看不到足夠背景

*分塊策略全景圖*

共有 5 種常見的分塊策略，各有優缺點：

| 策略名 | 原理說明 | 優點 | 缺點 | 適用場景 | 實施難度 |
|------|--------|-----|-----|---------|---------|
| **固定字符數** | 每 N 個字符截一次，可能重疊 | 快速、可預測、簡單 | 無視文檔結構，可能斷句 | 爬蟲數據、雜亂文本、快速原型 | ⭐ 簡單 |
| **固定 Token 數** | 用 tokenizer 計算精確 tokens，每 K 個 token 截一次 | 精確控制嵌入消費、兼容 LLM token 限制 | 需要 tokenizer，計算稍慢 | 面向 OpenAI API、需精確成本控制 | ⭐⭐ 中等 |
| **遞迴分割** | 先按 \n\n 分，再按 \n 分，再按 。 分，逐層遞迴 | **自適應保留文檔結構**，平衡精度 + 速度 | 配置參數多，調試複雜 | **推薦通用場景** | ⭐⭐ 中等 |
| **語言學分割** | 按句子、詞組層級分割，理解語義單位 | 完美保留語義邊界 | 演算法複雜、計算慢、需 NLP 模型 | 高精度需求、小規模知識庫 | ⭐⭐⭐ 困難 |
| **領域特定分割** | 按 Markdown 層級、代碼函數、JSON 欄位等結構分 | 完全尊重文檔邏輯結構 | 高度自訂，每個格式要單獨處理 | 特定格式的大型知識庫   | ⭐⭐⭐ 困難 |

*分塊參數的實務指南*

根據知識庫特性選擇合適的 chunk_size 和 overlap：

```
📋 場景 1：一般企業文檔（Word、PDF 文章）
chunk_size = 1000 字符 (~333 tokens)
overlap = 200 字符 (~67 tokens)
理由：平衡完整上下文與微粒檢索

📚 場景 2：技術文檔、編程教程、論文
chunk_size = 1500 字符 (~500 tokens)
overlap = 300 字符 (~100 tokens)
理由：需要更多上下文才能理解技術概念

🔬 場景 3：密集知識庫（法律、醫學、研究文檔）
chunk_size = 512 字符 (~170 tokens)
overlap = 100 字符 (~35 tokens)
理由：知識密度高，短文段就能表達完整概念，過長反而引入噪聲

💰 場景 4：成本敏感型（向量 DB 按 chunk 計費）
chunk_size = 2000 字符 (~667 tokens)
overlap = 100 字符 (~35 tokens)
理由：減少 chunk 數量降低成本，犧牲一些微粒度

⚡ 場景 5：實時系統（要求低延遲）
chunk_size = 800 字符 (~267 tokens)
overlap = 150 字符 (~50 tokens)
理由：較小的 chunks 搜索速度快，適合毫秒級 SLA

🔢 進階技巧：動態 chunk_size
可根據文檔複雜度動態調整：
- 簡單列表文檔：chunk_size = 512
- 中等複雜度文檔：chunk_size = 1000
- 高度專業文檔：chunk_size = 1500
```

*進階：Token 計數的數學*

你需要理解 token 計數為什麼不直觀——英文和中文的 token 效率完全不同：

```python
import tiktoken

# 英文例子
text_en = "Retrieval Augmented Generation improves LLM performance by integrating up-to-date information from external knowledge bases through dynamic retrieval mechanisms."
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokens_en = enc.encode(text_en)
print(f"英文: {len(text_en)} 字符 = {len(tokens_en)} tokens")
# 輸出: 166 字符 = 31 tokens (約 5.4 字符/token)

# 中文例子（意思相同）
text_zh = "通過檢索增強生成，LLM可以整合外部知識庫的實時信息，通過動態檢索機制增強性能。"
tokens_zh = enc.encode(text_zh)
print(f"中文: {len(text_zh)} 字符 = {len(tokens_zh)} tokens")
# 輸出: 38 字符 = 22 tokens (約 1.7 字符/token)

→ 結論：中文密度高得多！相同語義中文需要更少字符，但 token 數相近
→ 實踐建議：中文知識庫 chunk_size 可設得比英文大 2-3 倍
```

---

**構件 3：生成器（Generator）— RAG 系統的「大腦」**

*角色與決策*

生成器的工作很直接：**給定檢索到的上下文 + 用戶查詢，用 LLM 生成最終答案**。但選擇什麼 LLM、如何部署，涉及成本、品質、專遲多個維度的權衡。

*三大部署方案對比*

**方案 A：OpenAI API（推薦新手起步）**

特點：無需本地資源，開箱即用，API 調用式

模型選擇與成本分析：
```
GPT-3.5-turbo
  - Input 價量: $0.5 / 1M tokens (~200,000 個中文字，10 元)
  - Output 價量: $1.5 / 1M tokens
  - 實際成本估算：100 次查詢 × 平均 (800 input + 200 output) tokens
    = 100 × 800 × 0.0000005 + 100 × 200 × 0.0000015
    = 0.04 + 0.03 = $0.07（極便宜）
  - 延遲: 平均 2-3 秒（可接受大多數業務）
  - 品質：基線水平，適合 80% 應用

vs

GPT-4
  - Input 價量: $15 / 1M tokens (300 倍貴)
  - Output 價量: $45 / 1M tokens
  - 品質提升：+50% 準確性（但不是 15 倍）
  - 使用建議：只用於複雜推理、高價值業務
```

優點：
- 無需基礎設施投資
- API 極其穩定可靠
- 更新頻繁，能用上最新模型

缺點：
- 有成本（雖然不高）
- 依賴網絡連接
- 數據可能被 OpenAI 日誌記錄（對隱私敏感應用有風險）

---

**方案 B：本地開源模型（進階選項）**

特點：完全免費、私有、更新靈活，但需要一定技術成本

推薦組合：**Ollama + Mistral 7B**

為什麼 Mistral 7B？
- 參數量小（7B），在消費級 GPU（RTX 3050 8GB）上可跑
- 性能接近 GPT-3.5（很多基準測試相當或超過）
- 開源、可商用
- 指令微調優秀，適合 RAG

部署成本估算：
```
硬件：1 張 RTX 3050 (你已有)
電力：150W × 8 小時 = 1.2 kWh × $0.1/kWh = $0.12/天 (~$3.6/月)
總成本：舊 GPU 折舊 + 電費 ≈ $5/月 << $0.07/查詢 × 1000 查詢 = $70/月 (OpenAI)

性能：
Mistral 7B 推理速度 ~100-200 ms/查詢（比 OpenAI 快得多，因為是本地）

適合場景：內部企業系統、隱私敏感、查詢量大（邊際成本接近 0）
```

其他本地模型選項：
```
Llama 2 7B：最流行的開源模型，性能穩定
Neural Chat 7B：針對對話優化，比 Llama 好
Qwen 7B：開源中文模型，中文能力強
Gemma 7B：Google 的輕量級，效率高
```

缺點：
- 需要 GPU 硬件（你已有）、內存充足
- 部署和維護需要技術
- 模型質量可能低於 GPT-4 （但接近 GPT-3.5）
- 更新靠社區，可能不如商業模型及時

---

**方案 C：混合方案（實戰推薦）**

最優實踐：根據查詢複雜度動態選擇

```python
def select_generator(query):
    complexity_score = estimate_complexity(query)
    
    if complexity_score < 3:  # 簡單事實題
        return "mistral_7b_local"  # 更快、免費
    elif complexity_score < 5:  # 中等複雜度
        return "gpt-3.5-turbo"  # 平衡品質和成本
    else:  # 複雜推理題
        return "gpt-4"  # 最高品質

# 實例
select_generator("weight_decay 怎樣設置？")  # 簡單 → Mistral
select_generator("如何在不同數據集上選擇微調超參？")  # 中等 → GPT-3.5
select_generator("基於我正在進行的實驗，如何解決過擬合？")  # 複雜 → GPT-4
```

好處：
- 簡單查詢便宜、快速
- 複雜查詢保證質量
- 整體成本和性能達到平衡

---

##### ⚙️ 第三部分：簡單 RAG 與複雜 RAG 的架構差異

一旦你理解了 Retriever + Chunker + Generator 的基本工作流程，就可以開始思考如何優化系統。這正是簡單 RAG 與複雜 RAG 的區別所在。

**簡單 RAG：五步直線流程**

簡單 RAG 的工作流程是串聯的 5 個步驟：

```
[1] User Query
    ↓
    └─ 例：「微調中如何防止過擬合？」
    
[2] Query Embedding
    ↓ 用嵌入模型將查詢轉換為向量
    └─ "微調中如何防止過擬合？" → [向量 512 維]
    
[3] Vector Similarity Search
    ↓ 在向量庫中搜索最相似的 chunk
    └─ 返回 Top-5 chunks
    
[4] Prompt Assembly
    ↓ 把檢索結果組成 LLM Prompt
    └─ """基於以下文檔回答：
        {檢索到的 5 個 chunks}
        用戶問題：微調中如何防止過擬合？
        回答："""
    
[5] LLM Generation
    ↓ LLM 根據上下文生成答案
    └─ 「過擬合可通過 weight_decay、早停、數據增強...」
    
[6] End User
    ↓ 返回答案給用戶
    └─ 「微調防過擬合的三大技巧是...」
```

適用場景：
- 知識庫相對穩定，查詢相對直接
- 簡單事實題、常見 FAQ
- 不需要多輪推理

優勢：
- 實現簡單（一周內可完成原型）
- 可預測性強（容易進行性能分析）
- 部署快

缺點：
- 無法處理複雜查詢（「基於你的知識，有沒有其他方案？」）
- 無法檢測毫不相關的查詢（可能返回錯誤答案而非「無相關信息」）
- 無法進行多跳推理

---

**複雜 RAG：多層優化架構**

複雜 RAG 在簡單版本基礎上增加了多個優化層。讓我詳細說明：

```
[1] User Query: 「微調與 RAG 哪個更適合我的場景？」

[2A] Query Understanding & Rewriting（新增）
     ↓ LLM 進行 Query 改寫，使其更容易被檢索
     └─ 改寫：「微調的優缺點是什麼？RAG 的優缺點是什麼？」
     └─ 改寫：「微調和 RAG 的對比」
     └─ 生成 3 個不同變體，分別檢索（增加召回）

[2B] Query Expansion（新增）
     ↓ 補充關鍵詞
     └─ 原查詢：「微調」
     └─ 擴展：「微調」+ 「fine-tuning」+ 「參數更新」+ 「模型適配」

[3] Multi-Strategy Retrieval（改進）
     ├─ 策略 1：向量搜索（基於語義）
     │   └─ 返回 10 個 chunks
     ├─ 策略 2：BM25 搜索（精確關鍵詞匹配）
     │   └─ 返回 10 個 chunks
     └─ 結果合併去重
         └─ 共 15 個 candidates

[4] Re-ranking（新增）
     ↓ 用 CrossEncoder 對 15 個 candidates 重排
     └─ CrossEncoder 的評分更精確，會彙整考慮 Query + Chunk 的聯合相關性
     └─ 最終保留 Top-5

[5A] Multi-hop Retrieval（新增，用於複雜查詢）
     ↓ 識別查詢涉及多個主題
     └─ 第一跳：檢索「微調優缺點」
     └─ 第二跳：檢索「RAG 優缺點」
     └─ 第三跳：檢索「微調 vs RAG 對比」
     └─ 彙總 3 輪結果作為背景

[6] Prompt Assembly with Instructions（改進）
     ↓ 更精細的 Prompt 設計
     └─ """You are an expert assistant. Answer based strictly on the documents.
        If information is not in the documents, say so clearly.
        
        ## Documents:
        {15 個精選 chunks}
        
        ## User Query:
        微調與 RAG 哪個更適合我的場景？
        
        ## Answer Format:
        1. [微調適用場景]
        2. [RAG 適用場景]
        3. [混合方案]
        4. [我的建議]
        """

[7] LLM Generation with Chain-of-Thought（改進）
     └─ LLM 逐步說明推理過程
     └─ 「首先，微調適合...因為...」
     └─ 「其次，RAG 適合...因為...」
     └─ 「綜合考慮，我建議...」

[8] Output Verification（新增）
     ├─ 檢查引文是否真的在文檔中
     ├─ 檢查邏輯是否一致
     └─ 若檢驗失敗，提示用戶

[9] End User
     └─ 「根據你的場景，建議使用 RAG，因為...」
```

新增的優化點：
- Query Rewriting：改寫查詢使其更容易被檢索
- Multi-hop：對複雜查詢進行多輪檢索
- Re-ranking：重排檢索結果以提升相關性
- Prompt Engineering：更複雜的 Prompt 設計
- Output Verification：驗證輸出正確性

適用場景：
- 複雜開放式問題
- 需要推理和綜合多份文檔
- 對準確性要求極高

成本：增加 30-50% 計算時間和成本（換來 20-30% 準確性提升）

---

**本階段策略：先建基礎，後加優化**

根據經驗，最佳的學習路徑是：

1. **第 1-3 天：掌握簡單 RAG（當前階段）**
   - 完整理解 3 個構件
   - 能跑通完整流程
   - 度量 Recall、準確性等基本指標

2. **第 4-7 天：學習複雜 RAG 的一個優化**
   - 比如先學 Query Rewriting
   - 看看效果如何提升
   - 然後加入 Re-ranking

3. **第 8+ 天：多層優化（如果時間允許）
   - 結合多個優化技巧
   - 做系統性的 Ablation 實驗
   - 找到針對你的知識庫最有效的組合

不要一次全做，容易陷入「特性爆炸」而無法聚焦。簡單的東西做好，比複雜的東西都做不好更有價值。

---

#### 1.2 環境準備與知識庫構建實戰指南

##### 📦 RAG Python 環境從零到一

在開始動手前，需要構建一個完整的 Python 開發環境。環境搭建決定了 RAG 系統的穩定性、可達性和代碼質量。不含糊其邊的環境搭建往往能避免 80% 的隱性障礙。

**為什麼虛擬環境這麼重要？**

RAG 項目依賴多個相互關聯的包（langchain、faiss、openai 等），版本配置稍有不當就會產生難以調試的問題。虛擬環境通過隔離項目依賴來解決這一問題：
- 不同項目可能需要同一包的不同版本，全局安裝會導致版本衝突
- 隔離環境使項目可復現，便於團隊協作
- 卸載項目時只需刪除一個目錄，不污染系統環境

**五步驟環境配置完整指南**

- **詳細安裝步驟及常見問題排查**：
      ```bash
      # === 步驟 1：虛擬環境建立 ===
      python -m venv rag_venv
      
      # === 步驟 2：激活虛擬環境（根據系統選擇）===
      # Windows (PowerShell)
      .\rag_venv\Scripts\Activate.ps1
      # 如果遇到權限錯誤，執行：
      Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
      
      # Windows (Command Prompt)
      rag_venv\Scripts\activate
      
      # Linux/Mac
      source rag_venv/bin/activate
      
      # === 步驟 3：升級 pip（避免版本衝突）===
      python -m pip install --upgrade pip setuptools wheel
      
      # === 步驟 4：安裝核心依賴（推薦按順序）===
      # 第一組：基礎框架
      pip install langchain==0.1.20  # 指定版本避免不兼容
      pip install langchain-community==0.0.40
      pip install langchain-openai==0.1.15
      
      # 第二組：向量和 LLM
      pip install llama-index==0.9.50
      pip install faiss-cpu==1.7.4  # 或 faiss-gpu 如果有 NVIDIA GPU
      pip install openai==1.43.0
      
      # 第三組：工具包
      pip install pandas==2.0.0
      pip install numpy==1.24.0
      pip install python-dotenv==1.0.0
      pip install pyyaml==6.0
      
      # 第四組：可選（PDF 支援）
      pip install pypdf==4.0.0
      pip install pdf2image==1.16.3
      
      # === 步驟 5：全面驗證 ===
      python -c "
      import langchain; print(f'✓ langchain {langchain.__version__}')
      import faiss; print('✓ faiss-cpu')
      from langchain_openai import OpenAIEmbeddings; print('✓ langchain-openai')
      import llama_index; print('✓ llama-index')
      print('\n✅ 所有依賴安裝成功！')
      "
      ```
    - **常見安裝問題排查表**：
      | 錯誤 | 原因 | 解決方案 |
      |------|------|----------|
      | "ImportError: No module named langchain" | 未激活虛擬環境 | 再次運行 Activate 腳本 |
      | "pip wheel failed" | 依賴版本衝突 | `pip install --upgrade pip` 後重試 |
      | "FAISS: ImportError" | np.int 廢棄（numpy 1.24+） | `pip install numpy==1.23.5` |
      | "OpenAI API key not found" | 環境變量未設 | 見下方 API 密鑰配置 |
      | 「装 CUDA 版 faiss」建議 | 想用 GPU 加速 | `pip uninstall faiss-cpu && pip install faiss-gpu-cu11` |
    - **OpenAI API 密鑰配置**（3 種方式）：
      - 方式 1：環境變量（推薦）
        ```bash
        # 臨時設置（重啟後失效）
        $env:OPENAI_API_KEY="sk-xxxx..."
        
        # 永久設置（編輯系統環境變量）
        # 或在代碼中：
        import os
        os.environ["OPENAI_API_KEY"] = "sk-xxxx..."
        ```
      - 方式 2：.env 文件（最安全）
        ```
        # 在項目根目錄創建 .env
        OPENAI_API_KEY=sk-xxxx...
        
        # 在代碼中加載
        from dotenv import load_dotenv
        load_dotenv()
        ```
      - 方式 3：直接在代碼中傳遞
        ```python
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(api_key="sk-xxxx...")
        ```
    - **包功能對照說明**：
      | 包名 | 版本 | 核心功能 | 依賴項 | 備註 |
      |------|------|--------|--------|------|
      | **langchain** | 0.1.x | RAG 框架、Chain 組合、Prompt 管理 | 無 | 必須 |
      | **langchain-community** | 0.0.x | 第三方集成（Google、Hugging Face 等） | langchain | 推薦 |
      | **langchain-openai** | 0.1.x | OpenAI API 官方集成（比 openai 包更新） | langchain | 必須 |
      | **openai** | 1.4.x | 原生 OpenAI SDK（備用） | 無 | 備用 |
      | **faiss-cpu** | 1.7.x | 本地向量搜索（CPU 版） | numpy | 本地用必須 |
      | **faiss-gpu-cu11** | 1.7.x | 向量搜索 GPU 加速版 | CUDA 11.x | 可選 |
      | **llama-index** | 0.9.x | 數據索引、查詢引擎 | 無 | 推薦 |
      | **python-dotenv** | 1.0.x | 讀取 .env 環境文件 | 無 | 推薦（安全存密鑰） |
      | **pyyaml** | 6.0 | 配置文件解析 | 無 | 推薦 |
      | **pypdf** | 4.0.x | PDF 文本提取 | 無 | 處理 PDF 需要 |
      | **sentence-transformers** | 2.2.x | 本地嵌入模型 | torch | 不想用 OpenAI API 需要 |

##### 📚 第二部分：知識庫準備與本地文件加載體系

環境搭建好後，下一步是準備知識庫。知識庫的質量直接決定 RAG 系統的性能。「垃圾進垃圾出」這個原理在 RAG 系統上體現得最明顯。

**為什麼要用自己的文檔建立知識庫？**

很多初學者選擇使用公開數據集（如維基百科、新聞抓取集）進行測試。但這做法存在根本問題：
- 無法直觀體驗效果（不是在回答你熟悉的領域）
- 遠離真實應用場景（企業 RAG 系統通常面對私有文檔）
- 無法後續複用（公開數據集沒有後續價值，一次性消耗品）

相比之下，用自己的文檔搭建知識庫有三大優勢：
1. **即時反饋**：能快速判斷 RAG 是否真正理解了你的知識
2. **可後續複用**：搭建好的系統可以繼續應用在實際業務中
3. **貼近實戰**：所有問題和解決方案都是真實的，而非虛構場景

**推薦的知識庫文檔清單**

根據你的項目結構，以下文檔最適合作為初始知識庫素材：

```
優先級排序：
1. 微調深度解析.md              ← 你最熟悉的內容，深度足，最佳測試素材
2. 微調專項指南.md              ← 實踐指導，包含代碼示例
3. 學習與專案推進計畫.md        ← 提供項目上下文
4. LLM Course 課程筆記（如有）   ← 基礎概念參考
5. DistilBERT 微調經驗總結.md   ← 真實項目案例
```

為什麼選這些文檔？
- 完全是你親手寫的，質量和準確性有保證
- 涉及你最熟悉的領域（微調、LLM 優化），能快速驗證 RAG 的準確性
- 長度適中（不會太短導致無內容可檢索，也不會太長導致索引建立緩慢）
- 包含大量代碼片段，有利於測試「RAG 能否正確理解技術代碼」
- 覆蓋多個角度（概念、實踐、經驗），能全面測試檢索能力

**知識庫文件的質量檢查清單**

在把文件放入知識庫前，進行以下檢查以確保最佳質量：

| 檢查項 | 標準 | 為什麼重要 | 修復方案 |
|------|------|----------|--------|
| **文件編碼** | UTF-8 (中文檔必須) | 非 UTF-8 導致在加載時出現亂碼或 decode 錯誤，中斷整個流程 | VS Code 右下角選「UTF-8」，保存即可 |
| **文件大小** | 單個 < 10 MB，總計 < 100 MB | FAISS 全量載入內存，超大文件導致 OOM；API 有速率限制 | 大文件分割成多個小文件，或使用流式處理 |
| **文檔結構** | 有清晰的 Markdown 標題、段落劃分 | 結構清晰的文檔分塊效果好，最終檢索精度高 20-30% | 補充缺失的 Markdown 標題（如 # ## ###），拆分超長段落 |
| **語言一致性** | 主要用中文，避免大量英文混雜 | 混合語言降低嵌入模型的向量品質，造成語義理解偏差 | 決定知識庫的主要語言（本案例是中文），外文部分翻譯或隔離 |
| **特殊字符** | 避免過多特殊符號、表情符號、不可見字符 | 某些字符在 tokenization 步驟導致異常，破壞分塊邏輯 | 用正則表達式清理，或用文本編輯器的「查找替換」功能 |
| **重複內容** | 同一內容不應出現超過 2 次 | 重複內容導致向量庫中多個相同向量，浪費空間、計算資源，製造冗餘 | 手工去重或用 MD5 Hash 方式自動去重 |
| **敏感信息** | 不包含密碼、API key、個人隱私數據 | 洩露敏感信息導致安全風險，上傳雲端時風險加倍 | 用正則表達式掩蓋（如 `sk-proj-****...`），或直接刪除 |

**知識庫統計分析與預處理**

在正式開始向量化前，運行以下腳本來分析知識庫的統計特性，幫助你判斷知識庫是否適合進行 RAG：

```python
# 保存為 analyze_knowledge_base.py
import os
from pathlib import Path

def analyze_documents(directory="./knowledge_base"):
    \"\"\"分析知識庫文檔的統計信息\"\"\"
    documents = []
    total_chars = 0
    file_stats = []
    
    # 遍歷目錄中所有 .md 和 .txt 文件
    doc_dir = Path(directory)
    for file_path in doc_dir.rglob("*.md"):
        if file_path.is_file():
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                char_count = len(content)
                line_count = content.count("\\n")
                total_chars += char_count
                
                file_stats.append({
                    "file": file_path.name,
                    "chars": char_count,
                    "lines": line_count,
                    "avg_line_length": char_count / (line_count + 1)
                })
                documents.append(content)
    
    # 統計分析
    print(f"\\n📊 知識庫統計報告")
    print(f"=" * 60)
    print(f"文檔總數: {len(documents)}")
    print(f"總字符數: {total_chars:,}")
    if len(documents) > 0:
        print(f"平均文檔大小: {total_chars // len(documents):,} 字符")
    
    print(f"\\n單個文檔明細:")
    print(f"{'-' * 60}")
    
    for stat in sorted(file_stats, key=lambda x: x['chars'], reverse=True):
        print(f"  {stat['file']:30s} {stat['chars']:8,} 字 {stat['lines']:6,} 行")
    
    print(f"\\n📈 估計計算成本和資源需求:")
    print(f"{'-' * 60}")
    
    # 估計 chunks 數量和成本
    estimated_chunks = total_chars // 1000  # 假設預設 chunk_size=1000
    embedding_cost_input = (estimated_chunks * 0.0000005)  # OpenAI embedding-3-small 價格
    print(f"  預計 chunks 數: ~{estimated_chunks}")
    print(f"  OpenAI embedding 成本: ~${embedding_cost_input:.4f}")
    print(f"  建立 FAISS 索引時間: ~{max(1, estimated_chunks // 100)} 秒")
    print(f"  FAISS 索引記憶體占用: ~{estimated_chunks * 512 * 4 / 1024 / 1024:.1f} MB")
    
    return documents

if __name__ == "__main__":
    os.makedirs("./knowledge_base", exist_ok=True)
    analyze_documents()
```

運行此腳本後，你會得到類似這樣的輸出：
```
📊 知識庫統計報告
============================================================
文檔總數: 5
總字符數: 245,678
平均文檔大小: 49,136 字符

單個文檔明細:
  微調深度解析.md                79,234 字   1,245 行
  微調專項指南.md                65,123 字   1,023 行
  ...

📈 估計計算成本和資源需求:
  預計 chunks 數: ~246
  OpenAI embedding 成本: ~$0.000615
  建立 FAISS 索引時間: ~2 秒
  FAISS 索引記憶體占用: ~0.5 MB
```

**本地文本加載與驗證的完整代碼範本**

建立了知識庫文件夾後，使用以下代碼加載並驗證文檔：

```python
# 保存為 load_knowledge_base.py
from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader

def load_and_verify_documents(kb_directory="./knowledge_base"):
    \"\"\"加載並驗證知識庫文檔，確保質量\"\"\"
    
    print("📂 正在加載知識庫文檔...")
    
    # 使用 DirectoryLoader 加載所有 .md 文件
    loader = DirectoryLoader(
        path=kb_directory,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}  # 重要：指定中文編碼
    )
    
    try:
        documents = loader.load()
    except Exception as e:
        print(f"❌ 加載失敗: {e}")
        return None
    
    if not documents:
        print(f"❌ 在 {kb_directory} 中未找到任何 .md 文件")
        print(f"   提示：檢查目錄是否存在，文件是否為 .md 格式")
        return None
    
    print(f"✅ 成功加載 {len(documents)} 個文檔\\n")
    
    # 統計信息
    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"📊 統計信息:")
    print(f"  總字符數: {total_chars:,}")
    print(f"  平均文檔大小: {total_chars // len(documents):,} 字符")
    
    # 逐檔案詳細信息
    print(f"\\n📋 各文檔詳情:")
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get('source', 'Unknown')
        char_count = len(doc.page_content)
        preview = doc.page_content[:80].replace("\\n", " ")
        print(f"  {i}. {Path(source).name:30s} ({char_count:,} 字)")
        print(f"     預覽：「{preview}...」\\n")
    
    return documents

if __name__ == "__main__":
    kb_dir = "./knowledge_base"
    Path(kb_dir).mkdir(exist_ok=True)
    
    # 加載文檔
    docs = load_and_verify_documents(kb_dir)
    
    if docs:
        print("✅ 知識庫加載驗證成功！")
        print("   建議下一步：進行文本分塊（見 1.3 節）")
    else:
        print("❌ 加載失敗，請檢查知識庫目錄和文件格式")
```

成功執行此腳本後，你將看到一份完整的知識庫檔案報告，確認所有文件都正確加載。

---

#### 1.3 文本分塊 (Chunking) - 語義檢索的基石

文本分塊（Chunking）是 RAG 系統中最常被忽視卻至關重要的一環。並非所有文本都能直接被嵌入模型處理，也並非所有分塊策略都能帶來相同的效果。本節深入探討分塊的核心原理、科學設置方法，以及實戰中的最佳實踐。

##### 📖 第一部分：為什麼要分塊？四大核心作用

**1. 嵌入模型 Token 限制突破**

不同的嵌入模型都存在最大 token 限制。這是一個硬性上限：
- OpenAI text-embedding-3：最大 8191 tokens (~32KB 文本)
- 本地 sentence-transformers/all-MiniLM-L6-v2：512 tokens
- multilingual-e5-large：512 tokens

這意味著一篇 100KB 的論文無法直接嵌入。不是「可能失敗」，而是「必然失敗」。分塊就是把大文檔切成小段，確保每個片段都在模型的能力範圍內。

**2. 檢索精度提升 30-50%**

直觀理解：完整文檔嵌入會把所有細節「壓縮」到一個向量中。想象你用一個 1536 維的向量描述整本書——肯定會丟失大量信息。相比之下，只為相關段落編碼就清晰得多。實驗數據顯示，使用適當分塊的 RAG 系統檢索準確率比全文嵌入高 30-50%。

**3. 成本和效率控制**

從工程角度：100 個經過精心設計的小塊 vs 1 個大文檔的向量庫有根本性差異：
- 小塊：精確定位答案、快速檢索、易於增量更新
- 大塊：重複計算、搜索速度慢、更新困難（需重新嵌入整個文檔）

標準實踐中，chunk_size=1000 時的檢索成本只是 chunk_size=5000 時的 1/5。

**4. LLM 上下文窗口最大化**

RAG 的典型 Prompt 結構是：「基於以下文本回答：{chunks}\n\n問題：{query}」。如果 chunk 太大，組裝後很容易超過 LLM 的上下文限制。例如 GPT-3.5 只有 4K 上下文，大約能容納 3-4 個較大的 chunks；而 GPT-4 的 32K 版本可容納 8-10 個小 chunks，檢索覆蓋面更全面。

##### 📊 第二部分：Token 計數深度解析

Token 是嵌入模型世界的「貨幣」。精確理解 token 計數對設置參數至關重要。

```python
import tiktoken

# 不同語言的 token 長度對比
text_en = "Retrieval Augmented Generation is cool"
text_zh = "檢索增強生成很酷"

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
print(f"英文: {len(enc.encode(text_en))} tokens, {len(text_en)} 字符")
print(f"中文: {len(enc.encode(text_zh))} tokens, {len(text_zh)} 字符")

# 結果通常是：
# 英文: 7 tokens / 37 字符 → 平均 5.3 字符/token
# 中文: 11 tokens / 8 字符 → 平均 0.7 字符/token（中文更「重」）
```

**關鍵發現**：中文文本的 token 效率遠低於英文。中文是一門「奢侈的語言」——同樣的信息量需要更多 tokens。這直接影響分塊策略。

##### 🔧 第三部分：Chunk Size 的科學設置

不是所有 chunk_size 都相等。下表展示了不同 chunk_size 在真實場景中的表現：

| chunk_size | 優點 | 缺點 | 最佳應用場景 |
|-----------|------|------|-----------|
| **256** | 極度精細、成本低、精度高 | 上下文少、容易丟失信息、過度細碎 | 高精度需求、小知識庫、短文檔 |
| **512** | 精度適中、成本合理、平衡 | 中等精度損失 | 密集知識（代碼、論文、技術文檔） |
| **1000** | **通用推薦、最常用、充分的上下文** | 略有冗餘、檢索成本適中 | **大多數場景的起點——推薦此值開始實驗** |
| **2000** | 信息保留率高、上下文豐富 | 成本倍增、檢索可能不夠精確、容易超過 LLM 上下文 | 長篇文章、報告、線性敘述文本 |
| **4000+** | 上下文極豐富、最少丟失 | 檢索模糊（太多無關信息）、超過大多數 LLM 上下文限制、不符合 RAG 精神 | **通常不推薦** |

**實踐建議**：從 1000 開始作为起点。如果发现检索结果中有上下文丢失的迹象（答案片段不完整），则尝试 1500；如果发现检索噪声大（返回很多无关文档），则尝试 512。

##### 🎯 第四部分：Chunk Overlap 的科學設置

重疊（overlap）決定了相鄰 chunks 之間的銜接程度。設置不當的 overlap 會導致信息丟失或冗餘。

```
情景 1：文本連貫性強（報告、論文、文章）
→ overlap = chunk_size * 0.2 = 200（20% 重疊）
→ 目的：保證長句子不被硬生生截斷，相關概念跨越邊界時能被捕捉

情景 2：結構化文本（代碼、配置、 JSON）
→ overlap = chunk_size * 0.05-0.1 = 50-100（5-10% 重疊）
→ 目的：代碼有明確的邊界（函數、類），不需要高度銜接

情景 3：密集概念（教科書、法律文書、技術手冊）
→ overlap = chunk_size * 0.3-0.4 = 300-400（30-40% 重疊）
→ 目的：同一概念常跨越多個 chunks，高重疊確保核心概念被完整檢索

通用建議：overlap = chunk_size * 0.2（即 20% 重疊）
           →  chunk_size=1000 時，overlap 應設為 200
```

##### 💡 第五部分：三大分塊策略實戰對比

**策略 1：固定字數分塊（最簡單，推薦入門）**

```python
doc = "這是一個很長的文本..." # 假設 5000 字
chunk_size = 1000  # 字符數，非 tokens
overlap = 200
chunks = [doc[i:i+chunk_size] for i in range(0, len(doc), chunk_size-overlap)]

print(f"分塊數量: {len(chunks)}")
# 優點：簡單、快速、適合中文（因為中文計算字符而不是 tokens）
# 缺點：無法精確控制 token 數量、容易在句子中間截斷
```

**策略 2：Token 級別分塊（更精確，推薦生產環境）**

```python
import tiktoken

text = "長文本..."
chunk_size_tokens = 1000
overlap_tokens = 200

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokens = enc.encode(text)

chunks_tokens = []
for i in range(0, len(tokens), chunk_size_tokens - overlap_tokens):
    chunk = tokens[i:i+chunk_size_tokens]
    chunks_tokens.append(enc.decode(chunk))

# 優點：精確控制 token 數
# 缺點：需要安裝 tiktoken，計算稍慢
```

**策略 3：按結構分塊（適合結構化文檔——推薦優先使用！）**

- **Markdown 文檔**：按標題層級分割（## 分割比 ### 優先）
- **Python 代碼**：按函數/類分割，不要在程式碼邏輯中間截斷
- **JSON 結構**：按欄位或記錄分割
- **表格和列表**：保持整個表格/列表為一個 chunk

這種方式因為尊重文檔的邏輯結構，通常帶來最好的檢索效果。

##### 🛠️ 第六部分：完整代碼實現 & 最佳實踐

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Step 1: 加載文件
loader = TextLoader("knowledge_base.md", encoding="utf-8")
documents = loader.load()  # 返回 Document 對象列表

# Step 2: 配置分塊器（推薦：遞迴分塊器，保留結構）
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "，", " "],  # 遞迴嘗試優先順序
    chunk_size=1000,           # Token 或字符數（看分割器實現）
    chunk_overlap=200,         # 重疊數量
    length_function=len,       # 長度計算函數（默認是字符）
)

chunks = splitter.split_documents(documents)

# Step 3: 驗證結果
print(f"✓ 總分塊數: {len(chunks)}")
print(f"✓ 平均每個 chunk 大小: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} 字符")
print(f"✓ 最小 chunk: {min(len(c.page_content) for c in chunks)} 字符")
print(f"✓ 最大 chunk: {max(len(c.page_content) for c in chunks)} 字符")

# Step 4: 觀察樣本
print(f"\n範本 chunk（第 0 個）：")
print(chunks[0].page_content[:300] + "...")
print(f"元數據: {chunks[0].metadata}")

# Step 5: 實驗不同參數
print("\n\n=== 參數實驗 ===")
for chunk_size in [512, 1000, 1500]:
    for overlap in [int(chunk_size * 0.1), int(chunk_size * 0.2), int(chunk_size * 0.3)]:
        splitter_exp = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        chunks_exp = splitter_exp.split_documents(documents)
        avg_size = sum(len(c.page_content) for c in chunks_exp) / len(chunks_exp)
        print(f"chunk_size={chunk_size}, overlap={overlap:3d} → {len(chunks_exp):4d} chunks, 平均 {avg_size:.0f} 字符")
```

**調試技巧**：
1. **觀察邊界**：查看 chunks[i-1] 和 chunks[i] 的銜接處，確保沒有意外截斷
2. **統計分佈**：如果 chunk 大小差異大，考慮調整分割符優先順序
3. **質量檢查**：隨機抽取 10 個 chunk，手工驗證是否完整且有意義
4. **A/B 測試**：分別用不同參數分塊後，對相同查詢進行檢索對比

##### 🎓 實踐檢查清單

□ 已安裝 LangChain：`pip install langchain`
□ 已理解 token 計數（用 tiktoken 測試）
□ 已從 chunk_size=1000 開始實驗
□ 已驗證分塊效果（查看邊界、檢查覆蓋）
□ 已記錄實驗數據（不同參數下的性能差異）
□ 準備進入下一步：向量嵌入

---

#### 1.4 向量嵌入 & 本地向量庫 - 將文本轉化為可比較的數學表示

向量嵌入（Embeddings）是現代 RAG 系統的靈魂。它把文本轉化為高維空間中的點，使得語義相似的文本彼此靠近。本節深入講解嵌入的原理、工具選擇，以及完整的實現流程。

##### 📖 第一部分：什麼是向量嵌入？深度解析

**概念的直觀理解**

向量嵌入將文本（一維字符串）轉化為高維數值向量。例如：
- 文本：「什麼是 RAG？」
- 嵌入後：[0.12, -0.45, 0.89, ..., 0.34]（例如 384 維或 1536 維）

每個維度都代表文本的某個語義特徵，但這些特徵是由深度神經網絡自動學習的，通常無法被人類直接解釋。不過，我們可以理解它們的整體效果：**語義相似的文本會產生相似的向量**。

**距離度量：如何判斷相似性？**

嵌入只是「轉化」，核心是說「兩個向量有多相似」。常見的距離度量有：

1. **歐氏距離（L2 Distance）**
   $$d = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}$$
   - 直觀：向量空間中的直線距離
   - 特點：受維度影響，高維空間中所有點都相距很遠
   - 用途：某些特殊場景

2. **餘弦相似度（Cosine Similarity，推薦）**
   $$\cos(\theta) = \frac{A \cdot B}{|A||B|}$$
   - 值域：[-1, 1]，1 表示完全相同，0 表示無關，-1 表示完全相反
   - 特點：只看方向，不看大小，對歸一化向量友好
   - 用途：RAG 系統標準選擇

**第實戰選擇：常見嵌入模型的對比**

| 模型 | 維度 | 成本 | 質量 | 速度 | 適用場景 |
|-----|------|------|------|------|----------|
| **OpenAI text-embedding-3-small** | 512 | $0.02 per 1M tokens | ⭐⭐⭐⭐ | 中等 | 通用、需要高質量、願意付費 |
| **OpenAI text-embedding-3-large** | 1536 | $0.15 per 1M tokens | ⭐⭐⭐⭐⭐ | 中等 | 極端追求質量、成本無限制 |
| **HuggingFace all-MiniLM-L6-v2** | 384 | 免費 | ⭐⭐⭐ | 快速 | 本地部署、能接受質量下降 |
| **multilingual-e5-large** | 1024 | 免費 | ⭐⭐⭐⭐ | 中等 | 多語言支持、中英文混合 |
| **BAAI bge-small-zh** | 512 | 免費 | ⭐⭐⭐⭐ | 快速 | 純中文、高性能 |

**推薦策略**：如果你的 OpenAI API 額度充足，用 small 模型；如果要本地、快速且支持中文，用 multilingual-e5 或 bge-small-zh。

##### 🏗️ 第二部分：FAISS - 本地向量搜索的最佳方案

**什麼是 FAISS？**

FAISS（Facebook AI Similarity Search）是業界標準的開源向量搜索庫。它把高維向量組織成高效的索引結構，使得百萬級向量的相似度搜索可以在毫秒級完成。

**為什麼選擇 FAISS？**

| 特點 | 描述 |
|-----|------|
| ✅ **完全免費** | 無需支付 API 費用或許可費 |
| ✅ **完全本地** | 無網絡依賴，所有數據在本地，隱私有保障 |
| ✅ **極速搜索** | 百萬級向量毫秒級返回結果 |
| ✅ **簡單易用** | API 簡潔直觀 |
| ❌ **記憶體驅動** | 所有向量必須加載到 RAM（大規模時可能受限） |
| ❌ **更新複雜** | 添加新向量需要重建索引，不支持真正的「追加」 |

**FAISS 快速入門**

安裝：
```bash
# CPU 版本（推薦，免 GPU）
pip install faiss-cpu

# GPU 版本（如果有 GPU 且需要超高速）
pip install faiss-gpu
```

基本代碼示例：
```python
import faiss
import numpy as np

# Step 1：準備向量（e.g., 1000 個文檔，384 維）
embeddings = np.random.rand(1000, 384).astype('float32')

# Step 2：創建索引（用 L2 距離）
index = faiss.IndexFlatL2(384)
index.add(embeddings)

# Step 3：查詢（找最相似的 5 個）
query_embedding = np.random.rand(1, 384).astype('float32')
distances, indices = index.search(query_embedding, k=5)

print(f"最相似的 5 個文檔索引: {indices[0]}")  # [42, 157, 203, 89, 321]
print(f"距離（越小越相似）: {distances[0]}")   # [0.12, 0.34, 0.45, 0.51, 0.58]
```

**高級索引類型選擇**

FAISS 支持多種索引類型，權衡速度和準確性：

```python
# 方案 1：`IndexFlatL2`（精確但慢）
# 完全精確搜索，適合 < 100 萬向量
index = faiss.IndexFlatL2(384)

# 方案 2：`IndexIVFFlat`（快速，精度 90%）
# 量化優化，適合 > 100 萬向量，速度快 100 倍
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(384), 384, 100)
# 第三個參數 100 是「聚類中心數」，通常設為 sqrt(N)

# 方案 3：`IndexHNSW`（超快）
# 圖論方法，比 IVFFLAT 更快，適合極端場景
index = faiss.IndexHNSWFlat(384, 32)
```

對於 RAG 應用，**推薦用 IndexFlatL2** 作為起點（直到性能成瓶頸）。

##### 🔌 第三部分：完整端到端流程

從文檔加載到嵌入建立，再到測試檢索的完整代碼：

```python
from langchain.embeddings.openai import OpenAIEmbeddings
# 或：from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# ===== Step 1：加載文檔 + 分塊 =====
loader = TextLoader("knowledge_base.md", encoding="utf-8")
documents = loader.load()
print(f"✓ 已加載 {len(documents)} 個文檔")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(documents)
print(f"✓ 已分為 {len(chunks)} 個 chunks")

# ===== Step 2：創建嵌入模型 =====
# 方案 A：用 OpenAI API（推薦質量優先）
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 方案 B：用本地模型（推薦成本優先）
# from langchain.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/multilingual-e5-large"
# )

# ===== Step 3：建立向量庫（這一步會調用嵌入 API）=====
print("⏳ 正在嵌入文本並建立向量庫...")
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")
print("✓ 向量庫已保存到 ./faiss_index")

# ===== Step 4：測試檢索 =====
test_queries = [
    "什麼是 RAG？",
    "如何設置 chunk_size？",
    "FAISS 的優缺點是什麼？",
]

for query in test_queries:
    print(f"\n【查詢】{query}")
    results = vector_store.similarity_search(query, k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"  結果 #{i}（來源：{doc.metadata.get('source', 'unknown')}）")
        print(f"  內容預覽：{doc.page_content[:150]}...")

# ===== Step 5：後續加載（無需重新嵌入）=====
# 下次構建系統時，可以直接加載已有的向量庫
loaded_vector_store = FAISS.load_local("faiss_index", embeddings)
results = loaded_vector_store.similarity_search("RAG 是什麼？", k=5)
```

##### 🎓 實踐檢查清單

□ 理解了向量嵌入的基本原理（文本 → 高維向量）
□ 知道了距離度量的區別（歐氏 vs 餘弦相似度）
□ 選擇了合適的嵌入模型（OpenAI vs 本地開源）
□ 安裝了 FAISS：`pip install faiss-cpu`
□ 運行了完整的端到端代碼
□ 測試了檢索效果，觀察了返回結果
□ 保存和加載了向量庫
□ 準備進入下一步：生成集成（Prompt 工程）

---

#### 1.5 生成集成 (Generation) - LLM 調用與 Prompt 工程

到此為止，我們已經建立了一個功能完整的檢索系統：能加載文檔、分塊、嵌入、並檢索相關段落。但 RAG 的最後一環——**生成**——同樣至關重要。這一章探討如何選擇合適的 LLM、工程化 Prompt，以及組建完整的端到端 RAG Pipeline。

##### 📖 第一部分：LLM 選擇 - 付費 vs 免費的權衡

**OpenAI 方案（推薦新手及生產環境）**

OpenAI 是目前最成熟、穩定的 LLM API 提供者。
- 模型選擇：
  - `gpt-3.5-turbo`：$0.5-1.5 per 1M input tokens，速度快，質量中上
  - `gpt-4`：$15-30 per 1M input tokens，質量最高，成本 10 倍
  - `gpt-4o`（最新）：$5-15 per 1M tokens，性價比最優

- 優點：
  - 質量高，幻覺少
  - API 穩定，文檔完善
  - 無需本地 GPU，只需網路連接

- 缺點：
  - 需付費（每月 $10-100 預算）
  - 有延遲（平均 1-2 秒/請求）
  - 需聯網，隱私敏感數據不適合

推薦起點：gpt-3.5-turbo（性價比最優）

**本地開源方案（進階，追求成本 and 隱私）**

如果你要完全離線、無成本運行，可用本地 LLM：

- Ollama（一鍵部署）
  ```bash
  # 安裝 Ollama：ollama.ai
  ollama pull mistral          # 下載 Mistral 7B（推薦）
  ollama run mistral           # 啟動本地服務
  # 然後用 LangChain 連接 http://localhost:11434
  ```

- 優缺點：
  - ✅ 完全離線、無成本、數據完全安全
  - ✅ 可自定義訓練
  - ❌ 質量遠低於 GPT-3.5（中文理解較弱）
  - ❌ 需要 GPU（顯存 ≥ 8GB）
  - ❌ 部署複雜，故障排查困難

推薦起點：Mistral 7B（平衡質量和速度）

**選擇決策樹**

```
是否有 GPU?
├─ 否 → 用 OpenAI（唯一選擇）
└─ 是
   ├─ 能接受付費? 
   │  ├─ 是 → 用 gpt-3.5-turbo（最推薦）
   │  └─ 否 → 用本地 Mistral 或 Llama
   ├─ 數據涉及隱私?
   │  ├─ 是 → 用本地方案
   │  └─ 否 → 用 OpenAI
```

##### 🎯 第二部分：Prompt 工程 - 以及為什麼它至關重要

**為什麼 Prompt 工程很難？**

RAG 系統的 Prompt 需要平衡多個目標：
1. **精確度**：只基於檢索結果回答（不編造）
2. **完整性**：提供足夠的上下文讓 LLM 理解
3. **可溯源**：告訴用戶答案來自哪裡
4. **效率**：Token 數不能太多（控制成本）

**陷阱示例：為什麼「基礎 Prompt」會失敗？**

```
問題：基礎 Prompt
【Prompt】
回答以下問題。
文檔：{context}
問題：{question}

回答：

【結果】
LLM 很容易編造答案或基於世界知識而非文檔回答
幻覺率：30-40%
```

**改進方案：明確分界線 + 限制推理空間**

```
【改進 Prompt】
你是一個嚴格遵守事實的助手。請基於以下文檔回答用戶問題。

========== 文檔開始 ==========
{context}
========== 文檔結束 ==========

使用者問題：{question}

回答需求：
1. 只能基於上述文檔內容回答
2. 如果文檔中沒有相關信息，必須回答「文檔中未提及此信息」
3. 答案末尾必須附上引用的文檔名和段落位置
4. 嚴禁編造或推測文檔外的信息
5. 如果問題涉及多個子問題，逐一列舉回答

回答格式：
【回答】
[具體回答]

【引用來源】
- 來源：{source}
- 位置：{line_number}
- 引用段落：{excerpt}

【結果】
幻覺率：30-40% → 5-8%
準確性提升 3-5 倍
```

**Prompt 模板對比表：根據複雜度選擇**

| 模板類型 | 適用場景 | 效果 | Token 成本 | 推薦性 |
|--------|--------|------|---------|---------|
| **Zero-shot Plain** | 簡單事實題 | 50-60% 準確 | 基礎 | ⭐ |
| **Few-shot**（1-3 示例） | 需要格式參考 | 70-80% 準確 | +20% | ⭐⭐ |
| **Chain-of-Thought (CoT)** | 複雜推理 | 85-90% 準確 | +30% | ⭐⭐⭐ |
| **ReAct**（思考+行動） | 多步驟任務 | 90% 準確 | +50% | ⭐⭐ |
| **RAG + CoT 組合** | 檢索+推理 | **95%+ 準確** | +40% | **⭐⭐⭐⭐⭐** |

**推薦最優實踐：System Prompt + User Prompt 分離**

```python
from langchain.prompts import ChatPromptTemplate

# System Prompt：設定 LLM 的行為和角色
system_prompt = """你是一個 RAG 系統的文檔助手。你的核心職責是：

1. 【嚴格】基於提供的文檔回答問題，不基于世界知識推測
2. 【透明】明確標記每個答案的信息來源
3. 【誠實】如果文檔中沒有答案，說『無法從文檔中回答』
4. 【完整】如果答案需要多個文檔片段支撑，逐一列舉

你必须理解，用户对你的信任完全取决于你的诚实性。编造答案会摧毁系统的价值。
"""

# User Prompt：具體的任務
user_template = """
文檔內容
========
{context}
========

用戶問題：{question}

請根據上述文檔回答。
"""

# 組建 ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_template)
])

# 結果對比：
# - 沒有 System Prompt：LLM 準確性 70-75%
# - 有 System Prompt：LLM 準確性 82-87%
# 提升：10-15%
```

**Prompt 調試技巧（實戰工具箱）**

技巧 1：讓 LLM 「思考」（CoT - Chain of Thought）
```python
cot_prompt = """
請先按以下步驟回答：

步驟 1：理解問題
問題的核心是什麼？

步驟 2：搜索文檔
文檔中哪些部分相關？

步驟 3：分析和綜合
不同部分如何相互支持答案？

步驟 4：給出最終回答
基於上述分析，答案是？

答案：
"""
```
結果：準確率提升 10-20%

技巧 2：可信度評估
```python
confidence_prompt = """
在回答前，先自我評估確信度：
- 信息直接出自文檔 = 95%+ 確信
- 文檔提及但需轉述 = 60-80% 確信
- 文檔未提及 = 0% 確信（必須說『無法回答』）

如確信度 < 70%，改回答『文檔中信息不足』而不是猜測。
"""
```

技巧 3：Token 限制（降低成本）
```python
length_prompt = """
回答必須控制在 200 token 以內（約 150 字中文）。
優先包含最重要信息，次要信息可簡化或略去。
"""
```
成本降低：20-30%

##### 🏗️ 第三部分：完整 RAG Pipeline 代碼

從用戶查詢到生成答案的完整流程：

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# ===== 組件 1：初始化向量庫和 LLM =====
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local("faiss_index", embeddings)
# retriever：負責從向量庫檢索相關文檔
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # 返回 Top-5 相關文檔
)

# LLM：生成答案
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,  # 低溫度 = 更確定性、更少創新
    max_tokens=500
)

# ===== 組件 2：定義 Prompt 模板 =====
system_prompt = """你是一個幫助用戶基於文檔找答案的助手。

重要規則：
1. 只基於文檔內容回答，不推測或編造
2. 每個答案必須附上來源文檔和行號
3. 如文檔中沒有答案，說『文檔中未提及』而不是編造
"""

template = """{context}

用戶問題：{question}

請回答。"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", template)
])

# ===== 組件 3：組建 RAG Chain =====
# 這是一個 LangChain 的「可運行序列」(Runnable)
rag_chain = (
    {
        "context": retriever,        # 檢索相關文檔
        "question": RunnablePassthrough()  # 傳遞用戶問題
    }
    | prompt                         # 應用 Prompt 模板
    | llm                            # 調用 LLM
)

# ===== 組件 4：測試 =====
test_queries = [
    "什麼是 RAG？",
    "如何設置 chunk_size？",
    "FAISS 和 Milvus 有什麼差別？",
]

for query in test_queries:
    print(f"\n【用戶問題】{query}")
    print("-" * 50)
    
    response = rag_chain.invoke(query)
    
    print(f"【答案】")
    print(response.content)

# ===== 組件 5：進階用法 - 格式化輸出 =====
# 如果要返回結構化答案（答案 + 引用 + 權信度），可以這樣做：
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="answer", description="基於文檔的回答"),
    ResponseSchema(name="sources", description="引用的文檔名稱"),
    ResponseSchema(name="confidence", description="回答的可信度（0-100%）"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# 在 Prompt 中加入格式指令
template_with_format = template + f"\n\n{format_instructions}"

# 然後就能解析結構化輸出了
answer_dict = output_parser.parse(response.content)
```

##### 🎓 實踐檢查清單

□ 選擇了 LLM（OpenAI gpt-3.5-turbo 推薦作為起點）
□ 設置了 OPENAI_API_KEY 環境變數
□ 理解了 Prompt 工程的重要性
□ 測試了改進 Prompt 前後的效果差異
□ 實現了完整的 RAG Pipeline
□ 進行了至少 5 個査詢測試
□ 記錄了答案質量和幻覺率
□ 準備進入第二階段：檢索優化



---

### 第二階段：優化 & 評估（第 4-7 天）

目標：改進檢索品質、建立評估指標

#### 2.1 檢索優化 - 從「召回率」到「精準度」的進階之路

基礎 RAG 已經能工作，但檢索品質往往決定了整個系統的上限。這一章探討三大優化技術：混合檢索彌補單一方法的不足、Query 重構讓搜索更聰慧、Re-ranking 用精細排序確保最相關結果排在前面。

##### 📖 第一部分：混合檢索（Hybrid Search）- 彌補各自的不足

**向量搜索 vs BM25 的本質差異**

這是一個經典的「各有所長」場景：

**向量搜索（語義搜索）**
- ✅ 優勢：理解語義關係（「防止過擬合」≈「減少過度訓練」）、容忍表述差異、支持跨語言搜索
- ❌ 缺點：無法精確匹配專有名詞（不知道 Adam、weight_decay 的精確含義）、依賴模型質量、計算成本高
- 🎯 適用：概念類、知識類、抽象問題

**BM25（關鍵字搜索）**
- ✅ 優勢：精確匹配技術詞匯、代碼、公式；快速透明；在長文檔中穩定
- ❌ 缺點：無法理解語義（「降低損失」≠「最小化誤差」）、容易被高頻詞主導
- 🎯 適用：代碼片段、FAQ、文檔檢索

**混合檢索的威力**

```
單用向量搜索：
  查詢「Adam 優化器」→ 可能找到「動量優化」「梯度下降」（語義相近但不和匹配）
  
單用 BM25：
  查詢「Adam 優化器」→ 精確找到包含「Adam」的文檔
  
混合檢索：
  = 向量找「相似概念」+ BM25 找「精確關鍵詞」
  = 結果中既包含精確匹配，也包含語義相近
  = 召回率提升 30-50%，實際誤差率 ↓ 20%
```

##### 🔧 第二部分：三大融合策略詳解

**融合策略 A：加權融合（最簡單，推薦開始）**

```python
# score = λ × bm25_score + (1-λ) × vector_score
# λ 是權衡參數：
#   λ = 0.3：向量為主（常見概念査詢）
#   λ = 0.5：平衡（混合查詢）
#   λ = 0.7：BM25 為主（代碼/技術詞查詢）

from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(chunks)
vector_retriever = vector_store.as_retriever(search_kwargs={'k': 10})

# 創建混合檢索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # 70% 向量，30% BM25
)

results = ensemble_retriever.get_relevant_documents(query)
print(f"混合檢索返回 {len(results)} 個結果")
```

**融合策略 B：邏輯融合（精準但嚴格）**

交集模式（Intersection）：只返回「既語義相似 AND 關鍵詞匹配」的文檔
- ✅ 精準度高，幾乎沒有誤報
- ❌ 可能漏掉邊界情況

```python
vector_results = set(vector_retriever.get_relevant_documents(query)[:10])
bm25_results = set(bm25_retriever.get_relevant_documents(query)[:10])

# 交集：都滿足的文檔
intersection = vector_results & bm25_results
# 聯集：任一滿足的文檔
union = vector_results | bm25_results

print(f"向量搜索：{len(vector_results)} E，BM25：{len(bm25_results)} 篇")
print(f"交集（高精度）：{len(intersection)} 篇，聯集（高召回）：{len(union)} 篇")
```

**融合策略 C：級聯融合（速度 #1 推薦）**

先用快速檢索粗篩（BM25），再用精細排序（向量）

```python
# Step 1：快速粗篩（BM25），取 Top-20
initial_results = bm25_retriever.get_relevant_documents(query, k=20)

# Step 2：重排（向量嵌入 + 相似度排序），取 Top-5
query_embedding = embeddings.embed_query(query)

from sklearn.metrics.pairwise import cosine_similarity

scores = []
for doc in initial_results:
    doc_embedding = embeddings.embed_query(doc.page_content)
    similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
    scores.append((doc, similarity))

# Step 3：按相似度排序
reranked = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
final_results = [doc for doc, score in reranked]

print(f"✓ 從 {len(initial_results)} 篇縮小到 {len(final_results)} 篇相關度最高的")
```

**實戰決策樹：根據查詢類型調整策略**

```
查詢類型分類：
├─ 【事實型】「什麼是 X？」、「如何做 X？」
│  └─ 推薦策略：λ = 0.5（平衡），結果數 k = 10
│  └─ 理由：既需要概念理解又需要精確信息

├─ 【代碼型】「怎樣寫 Adam 優化」、「weight_decay 参数」
│  └─ 推薦策略：λ = 0.2-0.3（BM25 主導），結果數 k = 5
│  └─ 理由：代碼和參數名需精確匹配

├─ 【概念型】「為什麼 X 比 Y 好？」、「X 的本質是什麼？」
│  └─ 推薦策略：λ = 0.7-0.8（向量主導），結果數 k = 10
│  └─ 理由：概念理解更重要

└─ 【複雜型】「基於 X，如何做 Y 並評估 Z？」
   └─ 推薦策略：分解成 3 個子查詢，分別用不同 λ
   └─ 然後汇聚結果
```

##### 💡 第二部分：Query 重構（Query Rewriting）- 幫助你的搜索更聰慧

**為什麼需要重構？**

用戶的原始問題往往模糊、非正式、充滿上下文假設：
- 「那個電影怎樣？」（缺少電影名）
- 「類似的有過嗎？」（「類似」是模糊的）
- 「怎樣快速搞定這個？」（「這個」指什麼？）

知識庫需要更標準化的查詢語言，所以需要重構。

**重構 Prompt 示例**

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

rewrite_prompt = ChatPromptTemplate.from_template("""
你是一個查詢改寫助手。用戶提出的問題可能模糊或非正式。
請把它改寫成更適合在知識庫中搜索的形式，保留原意。

原問題：{question}

改寫要求：
1. 使用 3-5 個關鍵詞
2. 清晰、準確、標準化語言
3. 包含主要實體和動作

改寫結果：
""")

rewriter = rewrite_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# 測試
original = "那個電影怎樣？"
rewritten = rewriter.invoke({"question": original})
print(f"原問題：{original}")
print(f"改寫後：{rewritten.content}")
# 輸出可能：「電影評分、電影評價、推薦指數」

# 用改寫結果進行檢索
results = retriever.get_relevant_documents(rewritten.content)
```

**進階技巧：多查詢展開（Query Expansion）**

一個問題可能有多種表述方式，一次生成多個變體：

```python
expansion_prompt = ChatPromptTemplate.from_template("""
用戶想搜索：{question}

請生成 3 個不同角度的搜索查詢變體：
1. [角度 1]：
2. [角度 2]：
3. [角度 3]：

然後對每個變體進行搜索，最後合併結果。
""")
```

##### 🎯 第三部分：Re-ranking（重排） - 用精細排序確保最佳結果優先

**為什麼初步檢索還不夠？**

初步召回（Retrieval）召回 Top-20 可能存在：
- 相關度排序不完美（Top-10 中有幾篇邊界情況）
- 計算成本限制（全量用精細方法太昂貴）

解決方案：先快速召回，再用更精細的方法排序。

**CrossEncoder vs Bi-Encoder**

| 方面 | Bi-encoder（初步檢索） | CrossEncoder（重排） |
|-----|---------------------|---------------------|
| 方式 | Query → V1，Doc → V2，計算相似度 | [Query, Doc] → 聯合編碼 → 相關度分數 |
| 速度 | 快（向量可預計算） | 慢（每個查詢需重新計算） |
| 精度 | 70-80% | 90-95% |
| 成本 | 低 | 中等（只用於前 K 篇） |
| 推薦用途 | 初步召回 | 精細排序 |

**CrossEncoder 實現**

```python
from sentence_transformers import CrossEncoder
import numpy as np

# Step 1：加載預訓練 CrossEncoder 模型
# 推薦模型：
#   - 'cross-encoder/qnli-distilroberta-base'（通用，快速）
#   - 'cross-encoder/mmarco-mMiniLMv2-L12'（多語言，推薦中文）
cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12')

# Step 2：初步檢索（用向量或 BM25），取 Top-20
initial_results = retriever.get_relevant_documents(query, k=20)

# Step 3：準備 [Query, Doc] 對
pairs = [[query, doc.page_content] for doc in initial_results]

# Step 4：批量計算相關度分數（0-1）
scores = cross_encoder.predict(pairs)

# Step 5：重排並取 Top-5
ranked_with_scores = sorted(
    zip(initial_results, scores),
    key=lambda x: x[1],
    reverse=True
)[:5]

final_results = [doc for doc, score in ranked_with_scores]

print(f"初步檢索 {len(initial_results)} 篇 → 重排後 Top-{len(final_results)} 篇")
for i, (doc, score) in enumerate(ranked_with_scores, 1):
    print(f"  #{i}（分數 {score:.3f}）：{doc.page_content[:100]}...")
```

**完整優化 Pipeline**

```python
# = 混合檢索 + Query 重構 + Re-ranking 的完整組合 =

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from sentence_transformers import CrossEncoder

# 初始化
bm25_retriever = BM25Retriever.from_documents(chunks)
vector_retriever = vector_store.as_retriever(search_kwargs={'k': 15})
cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12')

# 混合檢索器
ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)

# Rewriter
rewriter_chain = rewrite_prompt | ChatOpenAI(...)

def optimized_retrieve(user_query):
    # Step 1：重構查詢
    rewritten = rewriter_chain.invoke({"question": user_query})
    search_query = rewritten.content
    
    # Step 2：混合檢索（取 Top-20）
    candidates = ensemble.get_relevant_documents(search_query, k=20)
    
    # Step 3：Re-rank（精排到 Top-5）
    pairs = [[search_query, doc.page_content] for doc in candidates]
    scores = cross_encoder.predict(pairs)
    
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]
    
    return [doc for doc, _ in ranked]

# 測試
results = optimized_retrieve("怎樣快速實現 RAG？")
```

##### 🎓 實踐檢查清單

□ 理解了向量搜索和 BM25 各自的優缺點
□ 實現了混合檢索（EnsembleRetriever）
□ 實驗了不同的權衡參數（λ）
□ 用 LLM 實現了 Query 重構
□ 安裝了 sentence-transformers：`pip install sentence-transformers`
□ 實現了 CrossEncoder 重排
□ 對比測試了「有無重排」的效果
□ 進行了至少 10 個查詢的檢索優化實驗
□ 記錄了優化前後的精度改進
□ 準備進入評估階段



---

#### 2.2 RAG 系統評估 - 從「能工作」到「有效工作」

RAG 系統構建完成後，如何判斷是否真的在工作？本章介紹系統評估的完整方法論，從關鍵指標定義、評估流程、到結果分析。

##### 📊 第一部分：核心評估指標詳解

**指標 1：檢索精度（Recall@K）**

定義：在 Top-K 結果中，相關文檔出現的比率。

$$\text{Recall@K} = \frac{\text{檢索到的相關文檔數}}{\text{該查詢的所有相關文檔數}}$$

為什麼用 Recall@5？因為 LLM Prompt 通常容納 5-10 個 chunks，第 5 份是實用邊界。

實例計算：
```
查詢：「如何防止過擬合？」
知識庫中所有相關文檔：
  1. regularization_techniques.md
  2. weight_decay_practice.md  
  3. overfitting_prevention.md
  → 共 3 篇相關文檔

系統返回的 Top-5：
  1. learning_rate_schedule.md (❌ 無關)
  2. weight_decay_practice.md (✅ 相關)
  3. batch_size_effects.md (❌ 無關)
  4. regularization_techniques.md (✅ 相關)
  5. activation_functions.md (❌ 無關)

Recall@5 = 2/3 ≈ 67%
相比目標 ≥ 80%，此系統檢索能力偏弱，需改進
```

**目標設定**：Recall@5 ≥ 80%（找到大多數相關文檔），Recall@10 ≥ 90%（進階目標）

**其他檢索指標對比表**

| 指標 | 計算方式 | 優勢 | 劣勢 | 何時使用 |
|-----|--------|------|------|---------|
| **Recall@K** | 相關文檔數 / 總相關數 | 直觀、易實現 | 忽視排序質量 | **推薦起點** |
| **MRR** | 1 / 第一相關結果排名 | 重視「找到第一個對的」 | 忽視其他結果 | 搜索引擎 |
| **NDCG@K** | 考慮排序 + 相關度等級 | 精細評估 | 計算複雜 | 推薦系統 |
| **MAP@K** | 逐K計算精度的平均 | 綜合評估 | 需多相關項 | 信息檢索 |

**指標 2：生成品質（Generation Quality）**

模型生成的答案是否正確、完整、且有根據？分為自動化和手工評估：

**人工評分（最可靠，最費時）**

評估維度：
- **準確性（Accuracy）**：答案是否絕對正確？
  - 0 = 完全錯誤或編造
  - 0.5 = 部分正確，有混淆或遺漏
  - 1 = 完全正確，符合知識庫

- **完整性（Completeness）**：是否涵蓋用戶期望的所有面向？
  - 1 分 = 極度不完整，遺漏關鍵信息
  - 3 分 = 基本完整，但可更詳細
  - 5 分 = 非常完整，涵蓋所有相關維度

- **相關性（Relevance）**：答案是否直接解決用戶問題？
  - 1 分 = 完全無關
  - 3 分 = 部分相關
  - 5 分 = 高度相關，解決核心問題

**自動化指標（粗略但快速）**

- **BLEU**：詞序匹配率（0-1，越高越好）
  - 優勢：計算快
  - 缺點：對 RAG 不友好，因為不同措辭（「降低損失」vs「最小化誤差」）會被評為不同

- **ROUGE**：召回率（0-1，越高越好）
  - 優勢：比 BLEU 對 RAG 友好
  - 缺點：仍不完美，語義可能相同但詞彙不同

- **BERTScore**：基於預訓練語言模型的語義相似度（0-1，推薦 ⭐⭐⭐⭐⭐）
  - 優勢：能理解語義相似，最適合評估 RAG 答案
  - 缺點：計算稍慢
  ```python
  from bert_score import score
  
  references = ["官方參考答案"]
  candidates = ["模型生成的答案"]
  
  P, R, F1 = score(candidates, references, lang="zh", verbose=False)
  print(f"BERTScore F1: {F1[0]:.3f}")  # 0-1，越高越好
  ```

**指標３：幻覺率（Hallucination Rate）**

模型是否編造了知識庫中不存在或不支持的信息？

幻覺檢測流程（SOP）：
1. 閱讀模型回答
2. 逐句提取事實聲明
3. 逐句查證知識庫（尋找明確支持）
4. 標記無根據部分

實例：
```
模型回答：
「微調時使用 Adam 優化器且 weight_decay=1e-4 可以有效防止過擬合。
 這是因為 Adam 的自適應學習率機制...」

事實檢驗：
✓ 「用 Adam 優化器」—— 知識庫有明確說明
✓ 「weight_decay=1e-4」—— 知識庫有推薦值
⚠️ 「Adam 的自適應學習率機制...」—— 知識庫未涉及，為幻覺

幻覺率 = 1 句 / 3 句 ≈ 33%
相比目標 < 10%，此系統需改進
```

目標：< 10%（越低越好；0% 是理想）

**其他關鍵指標**

- **相應時間（Latency）**：目標 < 3 秒（包括檢索 + LLM 推理）
- **單位成本（Cost）**：OpenAI API 約 $0.02-0.10 per request
- **用戶滿意度**：簡短問卷（1-5 分）或點讚率

##### 🛠️ 第二部分：實現簡單評估（10 分鐘快速版）

**第 1 步：準備測試集**

```python
test_cases = [
    {
        "question": "什麼是 RAG？",
        "expected_keywords": ["檢索", "增強", "生成"],
        "should_cite": ["RAG_intro.md"],
        "expected_answer_skeleton": "RAG 是一種 [...] 的技術"
    },
    {
        "question": "chunk_size 對性能的影響？",
        "expected_keywords": ["準確率", "速度", "權衡"],
        "should_cite": ["分塊策略.md"],
    },
    # 再加 8-10 個...
]
```

**第 2 步：運行系統並評估**

```python
import json

results = []

for test in test_cases:
    query = test["question"]
    
    # 運行 RAG
    response = rag_chain.invoke(query)
    retrieved_docs = retriever.get_relevant_documents(query, k=5)
    
    # 指標 1：檢索精度
    found_sources = [doc.metadata.get("source", "") for doc in retrieved_docs]
    retrieval_pass = any(
        source.endswith(cite) for cite in test["should_cite"] for source in found_sources
    )
    
    # 指標 2：關鍵詞覆蓋率
    answer = response.content
    keywords_found = sum(
        1 for kw in test["expected_keywords"]
        if kw in answer
    ) / len(test["expected_keywords"])
    
    # 指標 3：幻覺檢測（簡單）
    has_hallucination = any(
        suspect in answer for suspect in ["根據我的了解", "通常", "據說"]
    )
    
    results.append({
        "question": query,
        "retrieval_pass": retrieval_pass,
        "keywords_coverage": keywords_found,
        "no_hallucination": not has_hallucination,
        "response": answer[:300],
        "sources": found_sources
    })

# 第 3 步：統計結果
retrieval_accuracy = sum(r["retrieval_pass"] for r in results) / len(results)
keyword_coverage = sum(r["keywords_coverage"] for r in results) / len(results)
no_hallucination_rate = sum(r["no_hallucination"] for r in results) / len(results)

print(f"""
╔════════════════════════════════════════╗
║         RAG 系統評估報告               ║
╚════════════════════════════════════════╝

📊 核心指標
  • 檢索精度 (Recall@5): {retrieval_accuracy:.0%}   {'✅' if retrieval_accuracy >= 0.8 else '⚠️'}
  • 關鍵詞覆蓋: {keyword_coverage:.0%}      {'✅' if keyword_coverage >= 0.85 else '⚠️'}
  • 無幻覺率: {no_hallucination_rate:.0%}    {'✅' if no_hallucination_rate >= 0.9 else '⚠️'}

📝 評估結論
""")

if retrieval_accuracy < 0.8:
    print("  ⚠️ 檢索精度偏低，建議：改進 Query 重構 or 混合檢索配置")
if keyword_coverage < 0.85:
    print("  ⚠️ 答案完整性不足，建議：增加 retriever k 值 or 改進 Prompt")
if no_hallucination_rate < 0.9:
    print("  ⚠️ 幻覺率偏高，建議：改進 System Prompt or 訓練自定義重排器")
```

##### 📋 第三部分：完整評估報告模板

```markdown
# RAG 系統評估報告（日期：2024-01-15）

## 1. 測試集統計
- 總問題數：15
- 平均檢索精度 (Recall@5)：82%
- 平均幻覺率：7%
- 平均相應時間：2.1 秒
- 用戶滿意度評分：4.2 / 5.0

## 2. 指標達成情況
| 指標 | 目標 | 實際 | 狀態 |
|-----|------|------|------|
| Recall@5 | ≥ 80% | 82% | ✅ |
| 幻覺率 | < 10% | 7% | ✅ |
| 相應時間 | < 3s | 2.1s | ✅ |

## 3. 失敗案例分析

| 問題 | 失敗原因類型 | 具體失敗 | 改進方向 |
|-----|-----------|--------|--------|
| "如何選擇 chunk_size？" | 檢索不到 | 沒找到相關文檔 | 擴展知識庫 + Query 重構 |
| "LLM 原理是什麼？" | 內容混亂 | 答案缺乏邏輯 | 改進 Prompt 結構 |

## 4. 下一步優化計畫
1. 實施 Re-ranking 降低噪聲（預期 Recall 提升 5%)
2. 擴展知識庫 100+ 篇文檔
3. 實施用戶反饋迴圈
4. A/B 測試不同 Prompt 配置

## 5. 系統瓶頸診斷
- 主要瓶頸：檢索端（BM25 可能不適合該領域）
- 次要瓶頸：生成端 Prompt 工程（幻覺率）
- 推薦優先順序：先優化檢索 → 再優化生成
```

##### 🎓 實踐檢查清單

□ 理解了 Recall@K、MRR、NDCG 等檢索指標
□ 設計了 10-20 道測試問題
□ 運行了完整的系統評估
□ 記錄了檢索精度、幻覺率等關鍵指標
□ 識別了系統的主要瓶頸
□ 製作了評估報告
□ 準備進入第三階段：進階優化（可選）
      - 定義：在 Top-K 結果中，相關文檔的佔比 = 找到的相關文檔數 / 該查詢的所有相關文檔數
      - 為什麼用 Recall@5：LLM Prompt 通常容納 5-10 個 chunks，第 5 份是實用邊界
      - 實例計算：
        ```
        查詢：「微調如何防止過擬合？」
        知識庫中實際相關的文檔：
        1. regularization_techniques.md
        2. weight_decay_practice.md  
        3. overfitting_prevention.md
        
        系統返回的 Top-5：
        1. learning_rate_schedule.md (無關)
        2. weight_decay_practice.md (相關！)
        3. batch_size_effects.md (無關)
        4. regularization_techniques.md (相關！)
        5. activation_functions.md (無關)
        
        Recall@5 = 2/3 ≈ 67% (低於 80% 目標，需改進)
        ```
      - 目標：≥ 80%（找到大多數相關文檔；Recall@10 ≥ 90% 是進階目標）
    - **先進檢索指標對比**（MRR、NDCG、MAP）：
      | 指標 | 優勢 | 劣勢 | 何時使用 |
      |-----|------|-----|---------|
      | **Recall@K** | 直觀，易理解實現 | 忽略排序質量 | **推薦起點** |
      | **MRR** | 重視第一結果 | 忽視其他排序 | 搜索引擎評估 |
      | **NDCG@K** | 考慮排序 & 多級別相關性 | 計算複雜 | 推薦系統 |
      | **MAP@K** | 綜合評估全局相關性 | 需多相關文檔 | 信息檢索 |
    - **生成品質指標的詳細對比**：
      - 人工評分（最可靠，但最費時）：
        * 準確性 (Accuracy)：答案是否絕對正確？(0=完全錯誤 / 0.5=部分正確 / 1=完全正確)
        * 完整性 (Completeness)：是否涵蓋用戶期望的所有面向？(1-5 分)
        * 相關性 (Relevance)：答案是否直接解決用戶問題？(1-5 分)
      - 自動化指標（粗略但快速）：
        * BLEU：詞序匹配率（值域 0-1；但對 RAG 不友好，詞彙可能不同但語義相同）
        * ROUGE：召回率（值域 0-1；比 BLEU 好用，但仍有局限）
        * BERTScore：基於預訓練語言模型的語義相似度（推薦）
    - **幻覺率詳細檢測 SOP**：
      - 定義：模型生成了知識庫中不存在或不支持的信息
      - 檢測步驟（1. 閱讀模型回答 2. 逐句提取事實聲明 3. 逐句查證知識庫 4. 標記無根據部分）
      - 實例：
        ```
        模型回答：「微調時使用 Adam 優化器和 weight_decay=1e-4 可以有效防止過擬合，
                  這是因為 Adam 的自適應學習率可以...」
        
        事實檢驗：
        ✓ 「使用 Adam 優化器」—— 知識庫有明確說明
        ✓ 「weight_decay=1e-4」—— 知識庫有推薦值
        ✗ 「Adam 的自適應學習率可以...」—— 知識庫未涉及 Adam 算法原理
        
        幻覺率 = 1/3 ≈ 33% (高於 10% 目標，需改進)
        ```
      - 目標：< 10%（越低越好；0% 是理想狀態）
    - **其他關鍵性能指標**：
      - 相應時間 (Latency)：目標 < 3 秒（包括檢索 + LLM 推理）
      - 成本 (Cost)：按 API 調用估算（OpenAI $2-5/月 for 100k queries）
      - 用戶滿意度 (User Satisfaction)：簡單問卷（1-5 分）或點讚率

- [ ] **簡單評估方法**
  - [ ] 手工測試集：準備 10-20 個問題 + 預期答案
  - [ ] 逐題檢查：
    - ✅ 檢索到相關文檔？（是 / 否）
    - ✅ 答案是否基於文檔？（是 / 否）
    - ✅ 答案是否完整？（評分 1-5）
  - [ ] 計算準確率 & 幻覺率
  - 教材來源：中文教程 - 評估工具
  - 補充內容：
    ```python
    # 評估模板
    test_cases = [
        {
            "question": "什麼是 RAG？",
            "expected_keywords": ["檢索", "增強", "生成"],
            "should_cite": ["RAG_intro.md"]
        },
        {
            "question": "max_seq_length 對照實驗的結果如何？",
            "expected_keywords": ["256", "0.903", "最佳"],
            "should_cite": ["微調深度解析.md"]
        }
    ]
    
    results = []
    for test in test_cases:
        query = test["question"]
        
        # 運行 RAG
        response = rag_chain.invoke(query)
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # 檢查檢索
        retrieval_pass = any(
            doc.metadata.get("source", "").endswith(cite)
            for cite in test["should_cite"]
            for doc in retrieved_docs
        )
        
        # 檢查關鍵詞
        keywords_found = sum(
            1 for kw in test["expected_keywords"]
            if kw in response.content
        ) / len(test["expected_keywords"])
        
        results.append({
            "question": query,
            "retrieval_pass": retrieval_pass,
            "keywords_found": keywords_found,
            "response": response.content[:200]
        })
    
    # 統計
    recall = sum(r["retrieval_pass"] for r in results) / len(results)
    keyword_acc = sum(r["keywords_found"] for r in results) / len(results)
    print(f"檢索準確率: {recall:.0%}")
    print(f"關鍵詞匹配率: {keyword_acc:.0%}")
    ```

- [ ] **評估報告模板**
  - [ ] 記錄：測試集問題、檢索結果、生成回答、評估打分
  - [ ] 統計：平均檢索精度、幻覺率、用戶體驗評分
  - [ ] 瓶頸分析：哪類問題失敗最多？
  - 教材來源：自己設計
  - 補充內容：
    ```markdown
    # RAG 系統評估報告
    
    ## 測試集統計
    - 總問題數：15
    - 檢索精度 (Recall@5)：80%
    - 幻覺率：6%
    - 平均回答時間：2.3 秒
    
    ## 失敗案例分析
    
    | 問題 | 失敗原因 | 改進方向 |
    |-----|--------|--------|
    | "微調如何防止過擬合？" | 文檔中未直接提及 | 加入 weight_decay.md |
    | "LLM 原理是什麼？" | 檢索到無關文檔 | 改進 Query 重構 |
    
    ## 下一步優化
    1. 加入 Query 重構提升詞彙匹配
    2. 實施 Re-ranking 降低噪聲
    3. 擴展知識庫涵蓋更多主題
    ```

---

### 第三階段：進階 & 實踐（第 8-14 天，可選）

目標：優化系統、建立完整演示

#### 3.1 進階檢索技術

**多跳檢索（Multi-hop Retrieval）深度指南**

簡單檢索適合「單跳」問題（直接在知識庫中找答案）。但很多現實問題需要「多跳」：

成功案例：
- 「微調相比預訓練有什麼優勢，成本如何？」
  → 子問題 1：「微調的優勢」（檢索）
  → 子問題 2：「微調的成本」（檢索）
  → 合成：綜合兩個答案回答

- 「根據課程內容，下一步應該學什麼？」
  → 需要推理 + 檢索結合

**實現策略**：
```python
# Step 1：LLM 分解問題
from langchain.prompts import ChatPromptTemplate
decompose_prompt = ChatPromptTemplate.from_template("""
請把以下問題分解成 2-3 個獨立的子問題：
原問題：{question}

輸出格式：
1. 子問題 1：...
2. 子問題 2：...
""")

decomposer = decompose_prompt | llm

# Step 2：逐一檢索
all_docs = []
sub_questions = decomposer.invoke({"question": user_question}).content
for sub_q in sub_questions.split('\n'):
    if sub_q.strip():
        docs = retriever.get_relevant_documents(sub_q, k=3)
        all_docs.extend(docs)

# Step 3：去重 + 合成回答
unique_docs = list({doc.metadata['source']: doc for doc in all_docs}.values())
final_response = rag_chain.invoke({
    "context": unique_docs,
    "question": original_question
})
```

**知識圖譜 RAG（進階，超出時間預算）**

概念：不用純文本，而用結構化的知識圖譜（實體 + 關係）。

適用場景：
- 金融風險分析（實體：公司、風險類型；關係：「A 公司面臨 B 風險」）
- 醫療診斷（實體：症狀、疾病；關係：「症狀 X 可能指向疾病 Y」）
- 知識密集任務（需要多跳推理）

實現工具：
- **Neo4j**：開源圖數據庫（推薦）
- **LangChain KG 集成**：https://python.langchain.com/docs/modules/memory/kg/
- **LlamaIndex KG IndexRetriever**

**建議**：當知識庫擴大且需要複雜推理時再探索，時間有限可先跳過。

---

#### 3.2 工程化 & 部署

**RAG 系統模塊化架構**

將 RAG 分解為獨立模塊，每個模塊可獨立測試、替換、擴展：

```
Loader (讀取文件)
   ↓
Splitter (分塊)
   ↓
Embedder (嵌入)
   ↓
Retriever (檢索)
   ↓
Reranker (重排，可選)
   ↓
Generator (生成)
```

**用 LangChain Pipeline 實現**：

```python
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

# Step 1：定義各個模塊
def load_documents():
    loader = DirectoryLoader('knowledge_base', glob='**/*.md')
    return loader.load()

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# Step 2：組建 Pipeline
pipeline = (
    RunnableLambda(load_documents)
    | RunnableLambda(split_documents)
    | RunnableLambda(lambda chunks: create_embeddings(chunks))
    | RunnableLambda(lambda: build_retriever())
)

# Step 3：單元測試每個模塊
test_docs = load_documents()
test_chunks = split_documents(test_docs)
print(f"✓ 加載 {len(test_docs)} 個文檔，分成 {len(test_chunks)} 個 chunks")
```

**配置文件化**（YAML）：

```yaml
# config.yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"
  dimension: 512

retriever:
  type: "hybrid"  # "vector" / "hybrid" / "reranked"
  vector_weight: 0.6
  bm25_weight: 0.4
  top_k: 5

llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 2000

reranker:
  enabled: true
  model: "cross-encoder/mmarco-mMiniLMv2-L12"
  top_k_after: 5
```

使用配置文件：
```python
import yaml
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 根據配置動態構建系統
embeddings = OpenAIEmbeddings(model=config['embeddings']['model'])
```

**優勢**：
- 易測試：每個模塊可獨立驗證
- 易替換：切換嵌入模型無需改代碼，只需改 config
- 易擴展：新功能只需插入新模塊

**向量數據庫選擇（Pinecone vs FAISS）**

| 方面 | FAISS | Pinecone |
|-----|-------|----------|
| 存儲方式 | 內存（記憶體） | 雲端磁盤 |
| 適用規模 | < 1M 向量 | > 1M 向量 |
| 成本 | 免費 | 付費（$0.15+ per 1M vectors） |
| 維護 | 自管理 | 完全託管 |
| 實時更新 | 麻煩（需重建） | 容易 |

**何時遷移到 Pinecone**：
- 知識庫超過 1M 個 chunks
- 需要頻繁增刪改
- 願意為雲端託管付費

**遷移步驟**：
```python
# 從 FAISS 導出
index = faiss.read_index("faiss_index/index.faiss")
all_embeddings = index.reconstruct_n(0, index.ntotal)

# 上傳到 Pinecone
import pinecone
pc = pinecone.Pinecone(api_key="xxx")
pc.Index("rag-index").upsert(
    vectors=[(str(i), emb.tolist(), {"chunk_id": i}) 
             for i, emb in enumerate(all_embeddings)]
)
```

---

#### 3.3 效能評估與改進

**建立完整評估報告**

測試集設計（20+ 個多樣化問題）：
- 領域分布：RAG 概念、實現細節、故障排查各 1/3
- 難度分布：簡單（30%）、中等（50%）、困難（20%）
- 實際例子：
  ```
  簡單：「什麼是 chunk_size？」
  中等：「如何在混合檢索中平衡向量和 BM25 的權重？」
  困難：「系統檢索結果差，如何系統地排查？」
  ```

**完整評估報告結構**：

```markdown
# RAG 系統評估報告 v1.0（日期：2024-01-15）

## 1. 執行摘要
- 測試日期：2024-01-15
- 測試問題數：20
- 平均 Recall@5：82%
- 平均幻覺率：7%
- **系統評級：良好 (B)，可部署但須持續優化**

## 2. 測試集設計
- 知識庫覆蓋：RAG、微調、向量化等 5 個主題
- 問題難度分布：簡單 30%，中等 50%，困難 20%
- 參與評估者：1 名工程師（人工驗證）

## 3. 定量結果

### 檢索性能
| 指標 | 目標 | 實現 | 達成 |
|------|------|------|------|
| Recall@5 | ≥ 80% | 82% | ✅ |
| MRR@10 | ≥ 0.7 | 0.75 | ✅ |
| NDCG@5 | ≥ 0.65 | 0.68 | ✅ |

### 生成性能
| 指標 | 目標 | 實現 | 達成 |
|------|------|------|------|
| 幻覺率 | < 10% | 7% | ✅ |
| 準確性評分 | ≥ 4.0/5.0 | 4.2/5.0 | ✅ |
| 平均響應時間 | < 3s | 2.1s | ✅ |

## 4. 定性分析

### 成功案例（Recall = 100%）
- 「什麼是 RAG？」：完美檢索到 RAG_intro.md，答案準確
- 「chunk_size 怎樣設置？」：檢索并正確引用 chunking_guide.md

### 失敗案例（Recall < 50%）
| 問題 | 檢索精度 | 原因 | 改進方向 |
|------|---------|------|--------|
| 「LLM 原理是什麼？」 | 30% | 知識庫不涵蓋深層原理 | 補充 transformer.md |
| 「怎樣快速實現 RAG？」 | 40% | Query 表述模糊 | 加 Query 重構 |

## 5. 根本原因分析（RCA）

### 瓶頸 1：知識庫覆蓋不足（影響 30% 失敗）
- 根本原因：某些主題文檔缺失或不夠詳細
- 改進措施：補充 6-8 篇文檔

### 瓶頸 2：Query 表述多樣性（影響 20% 失敗）
- 根本原因：用戶表述和知識庫詞彙差異大
- 改進措施：實施 Query 重構層

### 瓶頸 3：檢索噪聲（影響 10% 失敗）
- 根本原因：無關文檔排在相關文檔前面
- 改進措施：加 Re-ranking

## 6. 下一步改進計畫（優先級排序）

1. **優先 P0**：補充知識庫（預期 Recall +10%）
   - 增加 6-8 篇關鍵主題文檔
   - 估計工作量：8 小時

2. **優先 P1**：實施 Re-ranking（預期 Recall +5%，幻覺率 -3%）
   - 集成 CrossEncoder
   - 估計工作量：2 小時

3. **優先 P2**：Query 重構（預期 Recall +3%）
   - 用 LLM 改寫查詢
   - 估計工作量：3 小時

## 7. 性能基準（Baseline）
- 簡單向量搜索：Recall 65%，幻覺率 12%
- 混合搜索（當前）：Recall 82%，幻覺率 7%
- 改進潛力：Recall 可達 92%，幻覺率可降至 < 5%
```

**人工推論驗證（Cherry Picking）**：

隨機抽 5-10 個結果，人工複查：
```python
# 檢查清單
checklist = {
    "事實正確": "是否有編造？",
    "有明確引用": "是否標明來源？",
    "回答完整": "是否涵蓋用戶期望的所有維度？",
    "語言流暢": "是否有語法錯誤？",
}

# 人工評分示例
result = {
    "query": "什麼是 RAG？",
    "response": "RAG 是...",
    "scores": {
        "fact_correct": 5,  # 1-5 分
        "citations": 4,
        "completeness": 5,
        "fluency": 5,
    },
    "notes": "優秀回答，清晰完整"
}
```

##### 🎓 實踐檢查清單

□ 實施了多跳檢索（可選）
□ 建立了模塊化 RAG 系統
□ 配置文件化了系統參數
□ 設計了 20+ 個測試問題
□ 完成了完整的評估報告（含數據、圖表）
□ 進行了人工驗證和失敗案例分析
□ 根據評估結果排優先級進行改進
□ 系統已可投入使用


---

## 📁 實踐項目結構（建議）

```
RAG_Project/
├── README.md                          # 項目說明、如何運行、已知限制
├── requirements.txt                   # pip install -r requirements.txt
├── config.yaml                        # 可配置的參數（模型、chunk_size 等）
│
├── 01_knowledge_base/                 # 知識庫文件（原始輸入）
│   ├── doc1.md
│   ├── doc2.md
│   ├── micro_tuning_guide.md
│   └── README.md                      # 文檔說明
│
├── 02_preprocessing/
│   ├── load_documents.py              # 讀取本地文件，輸出 LangChain Document
│   ├── text_splitter.py               # 文本分塊
│   └── preprocessed_chunks.pkl        # 序列化的 chunks（可選，加速重跑）
│
├── 03_embedding/
│   ├── create_embeddings.py           # 嵌入器初始化、文本→向量
│   ├── embeddings.pkl                 # 保存的向量（可選）
│   └── index_info.json                # 向量索引元數據（文檔名、chunk_id 等）
│
├── 04_retrieval/
│   ├── simple_retrieval.py            # 向量檢索（FAISS）
│   ├── hybrid_retrieval.py            # 混合檢索（向量+BM25）
│   ├── reranker.py                    # Re-ranking（CrossEncoder）
│   └── faiss_index/                   # FAISS 索引存儲目錄
│
├── 05_generation/
│   ├── basic_rag.py                   # 簡單 RAG Pipeline
│   ├── advanced_rag.py                # 優化版本（混合檢索+重排）
│   ├── prompt_templates.py            # Prompt 模板集中管理
│   └── utils.py                       # LLM 調用、格式化等工具函數
│
├── 06_evaluation/
│   ├── test_queries.json              # 手工編製的測試集（15+ 個問題）
│   ├── evaluate.py                    # 評估主腳本
│   ├── metrics.py                     # 評估指標計算
│   ├── results/                       # 評估結果
│   │   ├── evaluation_report.md       # 完整評估報告
│   │   ├── metrics.json               # 定量結果
│   │   └── error_analysis.txt         # 失敗案例分析
│   └── sample_outputs/                # 系統輸出樣例
│
├── 07_demo/
│   ├── cli_app.py                     # 命令行版本
│   ├── web_app.py                     # 簡單 Streamlit 版本（可選）
│   └── example_queries.txt            # 示範查詢集
│
└── docs/
    ├── setup_guide.md                 # 環境安裝步驟
    ├── architecture.md                # 系統設計說明
    └── learning_notes.md              # 學習筆記、坑點記錄
```

**關鍵文件說明**：
- `requirements.txt`：
  ```
  langchain>=0.1.0
  langchain-community
  langchain-openai
  llama-index
  faiss-cpu
  openai
  pandas
  numpy
  python-dotenv
  pyyaml
  ```

- `config.yaml` 示例：
  ```yaml
  # LLM 配置
  llm:
    provider: "openai"  # or "local"
    model: "gpt-3.5-turbo"
    temperature: 0.7
  
  # 嵌入配置
  embeddings:
    model: "text-embedding-3-small"
    dimension: 512  # 或 1536 for large
  
  # 檢索配置
  retrieval:
    type: "hybrid"  # "vector" / "hybrid" / "reranked"
    chunk_size: 1000
    chunk_overlap: 200
    top_k: 5
  ```

---

## ⏰ 時間預估

| 階段 | 主要任務 | 預計時間 | 交付成果 |
|------|---------|---------|--------|
| **第一階段** | 概念 + 基礎版本 RAG | 3-4 天（20-25 小時） | ✅ 能跑的簡單 Pipeline + 3 個測試 |
| **第二階段** | 混合檢索 + 評估系統 | 2-3 天（10-15 小時） | ✅ 完整評估報告（10+ 個測試案例） |
| **第三階段** | 進階優化（可選） | 2-4 天（10-20 小時） | ✅ Query 重構 + Re-rank + 20+ 測試 |
| **總計** | **完整 RAG 系統** | **5-7 天（30-40 小時）** | **到 4/1 前完成第一、二階段** |

---

## 📊 里程碑檢查清單

### 第一階段完成標誌
- [ ] 能用 LangChain 加載本地文檔
- [ ] 能用 FAISS 建立向量庫
- [ ] 能跑通簡單 RAG Pipeline（Query → 檢索 → LLM 生成）
- [ ] 測試 3 個問題，看結果是否合理
- [ ] 代碼放在 GitHub（或本地 git）

### 第二階段完成標誌
- [ ] 實現了混合檢索或 Query 重構
- [ ] 有完整的評估指標和報告
- [ ] 測試集 10+ 問題，統計檢索精度、幻覺率
- [ ] 對比改進前後的性能
- [ ] 瓶頸分析文檔完成

### 最終成果物
- [ ] ✅ 可跑的完整代碼（包含示例）
- [ ] ✅ 評估報告（含數據、圖表、結論）
- [ ] ✅ README（如何使用、已知限制、改進方向）
- [ ] ✅ 推論示例（5-10 個 Q&A）
- [ ] ✅ 學習筆記（關鍵坑點、設計決策記錄）

---

## 💡 學習心態提醒

1. **代碼優先於完美**：第一週的目標是「能跑」，不必代碼有多美麗
2. **邊學邊實驗**：遇到不懂的概念，直接在代碼中測試
3. **用自己的數據**：用個人文檔比公開數據集更有動力
4. **定期迭代**：每 2-3 天評估進度，及時調整方向
5. **接納不完美**：4/1 前完成第一、二階段即可，第三階段作為加分
6. **記錄過程**：每天簡單記一條「學到了什麼」、「遇到什麼坑」

---

## 🚨 常見坑點預警及深度排查指南

| 坑點 | 症狀 | 根本原因 | 初級排查 | 進階排查 | 預防方案 |
|------|------|--------|--------|--------|--------|
| **API 額度耗盡** | 突然 401 / 429 錯誤 + 服務中斷 | OpenAI 賬戶超額或被限流 | 檢查 OpenAI 官網賬戶頁，查看 Usage 和 Billing | 設置 API 金鑰過期時間；記錄每次 API 調用成本 | ✅ 預設 $5 月度預算；監控日誌 |
| **向量維度不匹配** | 「embedding size 不符」/ shape mismatch 錯誤 | 嵌入模型切換或版本不一致 | 檢查 embedding model、確認 embedding 維度 | 比對保存的向量與新模型維度；檢查 FAISS index | ✅ 鎖定嵌入模型版本；序列化元數據 |
| **檢索結果差** | Top-5 全是無關文檔，Recall < 50% | chunk_size 過大、相似度門檻不當、或文檔不夠相關 | 用簡單例子測試；檢查 chunk_size 和 overlap | 調整相似度門檻（0.6 → 0.5）；測試不同 embedding 模型 | ✅ 用 human-eval 測試集驗證；混合檢索作保險 |
| **幻覺嚴重** | 答案包含文檔中沒有的信息，幻覺率 > 20% | LLM 無法區分「文檔」和「知識庫」| 檢查 Prompt 是否清楚標記文檔邊界；抽查 5-10 條回答 | 實施 Re-ranking；用 Chain-of-Thought Prompt | ✅ 使用「必須引用」+「無法回答」的 Prompt |
| **性能慢** | 單次查詢 > 5 秒，用戶等待焦慮 | 向量計算慢 / LLM API 延遲 / 網絡問題 | 分別計時：檢索 vs LLM；檢查網絡延遲 | 改用 FAISS GPU 版；批量查詢；多線程 | ✅ 本地緩存熱點查詢；異步調用 |
| **FAISS 索引崩潰** | 「can't allocate memory」/ 索引文件損壞 | 向量太多或 RAM 不足；文件損壞 | 檢查系統 RAM；檢查索引文件大小 | 改用 FAISS GPU  版或壓縮量化；備份索引 | ✅ 定期備份 FAISS 索引；監控 RAM 使用 |
| **中文編碼問題** | 亂碼、特殊字符消失 | 文件編碼不統一（UTF-8 vs GBK） | 檢查文件編碼；確保 Python 使用 UTF-8 | 用 chardet 自動偵測編碼；統一轉換 | ✅ 所有文件統一 UTF-8；文檔首行標記編碼 |
| **Query 理解差** | 「那個東西怎樣」這類模糊問題無答案 | 依賴用戶提問精度 | 添加 Query 重構層；記錄失敗的 Query 模式 | 用 LLM 改寫後再搜；構建 Query 模板庫 | ✅ 提供查詢示例；建立常見問題 FAQ |
| **知識更新慢** | 最新文件加入後仍搜不到 | 沒有重建 FAISS 索引 | 檢查知識庫文件修改時間；重建一次索引 | 實施增量索引而非全量重建 | ✅ 自動化：監控文件夾，改動自動更新 |

**深度排查流程圖**：
```
遇到問題
  ↓
[症狀分類] → 速查表找對應症狀
  ↓
[初級排查] → 按「初級排查」欄逐步執行
  ├─ 解決 √ → 記錄成果 & 迭代
  └─ 未解決 ↓
[進階排查] → 按「進階排查」欄執行
  ├─ 解決 √ → 歸納根本原因
  └─ 未解決 ↓
[預防對策] → 採取「預防方案」欄的措施
  ↓
[文檔 & 分享] → 在團隊 Wiki 記錄此陷阱
```

**實戰案例 1：檢索結果差的完整排查**：
```
症狀：查詢「什麼是 weight decay？」返回的 Top-5 中有 3 個無關文檔
步驟 1：確認知識庫確實包含 weight_decay.md
步驟 2：測試向量相似度
  - 查詢向量化後與 weight_decay.md 的相似度 = 0.45（低於 0.6 門檻）
  - 原因：embedding 模型質量不佳或查詢表述差異大
步驟 3：嘗試 Query 重構
  - 原始：「什麼是 weight decay？」
  - 改寫後：「weight decay L2 正則化參數懲罰目錄 regularization」
  - 新相似度 = 0.72！
步驟 4：實施混合檢索
  - 向量 Top-5 + BM25 Top-5 融合
  - 結果大幅改善
結論：Query 表述和混合搜索是關鍵
```

**實戰案例 2：幻覺率高的排查與修復**：
```
症狀：系統回答「Adam 優化器使用自適應學習率可有效防止過擬合」
      → 知識庫未涉及 Adam 原理，純幻覺
步驟 1：識別幻覺特徵
  - 該陳述不在任何檢索到的文檔中
  - LLM 自我發揮超出文檔範圍
步驟 2：檢查 Prompt
  - 原 Prompt：「基於以下文檔回答：{context}」（太寬鬆）
  - 改進 Prompt：「只能使用以下文檔回答，如無相關信息必須說『文檔中未提及』」
步驟 3：實施 Re-ranking 確保相關性
  - 加入 CrossEncoder 過濾不相關文檔
  - 相關性 score < 0.5 的文檔被排除
步驟 4：評估改進
  - 平均幻覺率：33% → 8%（大幅改善）
結論：Prompt 措辭和 Re-ranking 雙管齊下效果最佳
```

---

## 💡 學習心態提醒 & 進階建議

### 核心心態
1. **代碼優先於完美**：第一週的目標是「能跑」，不必代碼有多美麗
2. **邊學邊實驗**：遇到不懂的概念，直接在代碼中測試（比紙上談兵快 10 倍）
3. **用自己的數據**：用個人文檔比公開數據集更有動力、業務場景也更貼近
4. **定期迭代**：每 2-3 天評估進度，及時調整方向（不要死鑽）
5. **接納不完美**：4/1 前完成第一、二階段即可，第三階段作為加分
6. **記錄過程**：每天簡單記一條「學到了什麼」、「遇到什麼坑」（建立自己的知識庫）

### 進階學習建議
- 第 1-3 天：專注「能跑」，不優化
- 第 4-5 天：逐項測試優化（Query 重構 → Mixed 檢索 → Re-ranking）
- 第 6-7 天：完整評估報告 + 對比實驗
- 第 8+ 天：根據評估結果逐一改進（優先級排序，ROI 導向）

---

## 📌 重要提示

- 這份計畫**靈活調整**，優先完成「基礎概念 + 最小版本」
- 如果某章節理解困難，可先**跳過**，用代碼實踐學習（做中學）
- 遇到卡點可**回到教材**、或**提出來討論**
- 評估報告和代碼一樣重要，別到最後才匆匆補
- 把失敗案例記下來，這些是改進方向最好的提示

---

## 🔗 速查表 & 實用資源

### 環境檢查命令
```bash
# 檢查 Python 版本
python --version  # 應為 3.9+

# 檢查 venv 是否激活
which python  # 或 Get-Command python (PowerShell)

# 檢查所有依賴包版本
pip list

# 檢查特定包版本
pip show langchain

# 導出當前環境配置（便於重建）
pip freeze > requirements.txt

# 檢查 OpenAI API 連接
python -c "import openai; print(f'OpenAI Version: {openai.__version__}')"
```

### 核心功能測試命令
```bash
# 檢查 LangChain 是否安裝
python -c "import langchain; print(f'✓ LangChain {langchain.__version__}')"

# 檢查 FAISS 是否可用
python -c "import faiss; print(f'✓ FAISS OK, CPU: {faiss.get_num_threads()} threads')"

# 檢查 Embeddings 是否工作
python -c "
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
result = embeddings.embed_query('test')
print(f'✓ Embeddings OK, dimension: {len(result)}')
"

# 運行評估腳本
python 06_evaluation/evaluate.py --config config.yaml --verbose

# 查看 FAISS 索引統計
python -c "
import faiss
idx = faiss.read_index('04_retrieval/faiss_index/index.faiss')
print(f'向量總數: {idx.ntotal}')
print(f'向量維度: {idx.d}')
"

# 匯出評估結果為 JSON
python 06_evaluation/evaluate.py --config config.yaml --output results.json

# 測試本地 Ollama 模型連接
python -c "
from ollama import Client
client = Client(host='http://localhost:11434')
response = client.generate(model='mistral', prompt='Hello')
print(f'✓ Ollama 連接成功')
"
```

### 快速除錯命令集
```bash
# 檢查知識庫文件編碼是否統一（UTF-8）
file *.md  # Linux/Mac

# 統計知識庫文件大小和字數
wc -c *.md  # 字節數
wc -w *.md  # 字數

# 查看最近修改的文件（用於檢查知識庫是否更新）
ls -lt 01_knowledge_base/ | head -10

# 測試單個 chunk 的嵌入
python -c "
from langchain_openai import OpenAIEmbeddings
e = OpenAIEmbeddings()
emb = e.embed_query('微調是什麼？')
print(f'Embedding 長度: {len(emb)}')
print(f'第一個值: {emb[0]:.6f}')
"

# 快速檢查 Prompt 模板是否正確
python -c "
from 05_generation.prompt_templates import RAG_PROMPT_TEMPLATE
print(RAG_PROMPT_TEMPLATE.template)
"
```

### 實用 Python 片段 (copy-paste ready)
```python
# === 片段 1：快速加載知識庫 ===
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    'knowledge_base',
    glob='**/*.md',
    loader_cls=TextLoader
)
docs = loader.load()
print(f'加載文檔數: {len(docs)}')

# === 片段 2：快速測試嵌入模型 ===
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
query_emb = embeddings.embed_query('什麼是 RAG？')
print(f'查詢嵌入維度: {len(query_emb)}')

# === 片段 3：快速建立向量庫 ===
from langchain.vectorstores import FAISS
vector_db = FAISS.from_documents(docs, embeddings)
vector_db.save_local('faiss_index')
print('✓ 向量庫已保存')

# === 片段 4：快速測試檢索 ===
retriever = vector_db.as_retriever(search_kwargs={'k': 5})
results = retriever.get_relevant_documents('微調最佳實踐')
for r in results[:3]:
    print(f'- {r.metadata.get(\"source\", \"Unknown\")} : {r.page_content[:100]}')

# === 片段 5：快速測試 LLM ===
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7)
response = llm.invoke('你是誰？')
print(response.content)
```

### 文檔快速連結 & 學習資源

**官方文檔**：
- 🔗 [LangChain 完整文檔](https://python.langchain.com/docs/)
  - RAG 專題：https://python.langchain.com/docs/tutorials/rag/
  - Chat Models：https://python.langchain.com/docs/integrations/chat/
- 🔗 [LlamaIndex 官方](https://docs.llamaindex.ai/)
  - 核心概念：https://docs.llamaindex.ai/en/stable/concepts/
- 🔗 [OpenAI API 文檔](https://platform.openai.com/docs/)
  - Embeddings：https://platform.openai.com/docs/guides/embeddings
  - Chat Completions：https://platform.openai.com/docs/guides/gpt

**工具庫文檔**：
- 🔗 [FAISS 官方](https://facebook.github.io/faiss/)
  - Quick Start：https://github.com/facebookresearch/faiss/wiki/Getting-started
- 🔗 [Sentence Transformers](https://www.sbert.net/)
  - 預訓練模型列表：https://www.sbert.net/docs/sentence_transformers/pretrained_models.html
- 🔗 [ChromaDB](https://docs.trychroma.com/) (FAISS 的替代品)

**深度學習資源**：
- 📚 [Transformer 架構圖解](http://jalammar.github.io/illustrated-transformer/)
- 📚 [向量檢索綜述論文](https://arxiv.org/abs/2305.03039)
- 📚 [RAG 系統綜述](https://arxiv.org/abs/2312.10997)

### 性能優化參考值
```
├─ Embedding 成本：$0.02 per 1M tokens (GPT-3 small)
├─ LLM 推理：0.5-2 秒/查詢 (API)，取決於模型大小和負載
├─ 檢索速度：< 100ms (FAISS on CPU with 100k vectors)
├─ Top-K 值：通常 5-10，大於 20 收益遞減
├─ Chunk 大小：推薦 800-1200 tokens
├─ Overlap：推薦 20-30% (160-360 tokens)
└─ Re-ranking 成本 vs 效益：+50% 延遲, +5-10% 準確度提升
```

### 常見問題速查
```
Q: 用 OpenAI 還是本地模型？
A: 開發用 OpenAI (GPT-3.5)，生產環境考慮本地 (Mistral 7B)

Q: chunk_size 和 overlap 怎麼設？
A: 起點是 1000/200，然後根據 eval 結果調整

Q: 什麼時候加 Re-ranking？
A: 當 Recall < 70% 或幻覺率 > 15% 時值得試試

Q: 中文支持怎麼樣？
A: multilingual-e5-large 最佳，text-embedding-3 也夠用

Q: 怎樣降低 API 成本？
A: 批量查詢、用本地 embedding 模型、減少 chunk 數量調試

Q: 為什麼檢索結果突然變差？
A: 最常見的原因：(1) 知識庫更新沒重建索引 (2) embedding 模型版本變了 (3) 相似度門檻被調動
```

---

## 🔌 RAG 實踐完全指南

為確保不留知識盲點，本章詳盡補充了從開發到生產的 4 大實踐領域。

### 4.1 上下文窗口管理（Context Window Optimization）

**什麼是上下文窗口限制？**

LLM 有 Token 限制：GPT-3.5-turbo 是 4K/16K Token，GPT-4 是 8K/128K Token。一個「完整的 Prompt」包括：

```
System Prompt（500 Token）
  + 檢索到的文檔（2000-4000 Token）  ← 這是你能控制的
  + 用戶問題（100 Token）
  + 生成的答案（2000 Token）
  = 總計：~4600 Token，超出 4K 限制！
```

**為什麼不能盲目塞入所有檢索結果？**

1. 成本線性增長：多 1K Token = 多支付 2-4 倍費用
2. 模型性能下降：超過 70% 窗口使用率時，幻覺率滑升 20-30%
3. 延遲增加：更多 Token = 更慢的推理

**實戰策略：三層過濾器**

```python
# 層級 1：檢索端過濾（減少初始結果）
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 只取 Top-3，不是 Top-10！
)

# 層級 2：Re-ranking 過濾（用 CrossEncoder 精選）
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
reranker = HuggingFaceCrossEncoder(
    model_name="cross-encoder/mmarco-mMiniLMv2-L12"
)

# Re-rank only keep top 2
ranked_docs = reranker.rank(
    query=question,
    corpus=[doc.page_content for doc in docs],
    top_k=2  # 精減至 2 份文檔
)

# 層級 3：內容過濾（刪除低信息密度的部分）
def filter_doc_content(doc_text, max_length=500):
    lines = doc_text.split('\n')
    # 刪除 code block（通常不相關）
    filtered = [l for l in lines if not l.strip().startswith('```')]
    # 按密度排序（優先保留有 keywords 的句子）
    return '\n'.join(filtered)[:max_length]
```

**Token 預算計算**

```python
from langchain.callbacks import get_openai_token_counter

def estimate_tokens(question, retrieved_docs, system_prompt):
    prompt_template = f"{system_prompt}\n\n"
    
    # 計算文檔部分
    doc_text = "\n---\n".join([d.page_content for d in retrieved_docs])
    prompt_template += f"文檔：{doc_text}\n"
    
    # 計算問題部分
    prompt_template += f"問題：{question}"
    
    counter = get_openai_token_counter("gpt-3.5-turbo")
    total_tokens = counter(prompt_template)
    
    # 預留 2000 Token 給 LLM 回答
    safe_limit = 3000  # 4K 總限制 - 1K 安全邊界
    
    if total_tokens > safe_limit:
        print(f"⚠️ 超出預算！{total_tokens} > {safe_limit}")
        print(f"   需要減少到 {safe_limit - total_tokens} Token")
        return False
    
    return True
```

**現實數據**

| LLM | 窗口大小 | 推薦文檔數 | 推薦文檔字數 |
|-----|---------|---------|----------|
| GPT-3.5 4K | 4000 | 2-3 篇 | 1500-2000 字 |
| GPT-3.5 16K | 16000 | 5-8 篇 | 5000-8000 字 |
| GPT-4 8K | 8000 | 3-4 篇 | 2500-3500 字 |
| GPT-4 128K | 128000 | 50+ 篇 | 50000+ 字 |
| Claude 3 200K | 200000 | 100+ 篇 | 100000+ 字 |

---

### 4.2 成本管理與優化

**成本構成與計算**

RAG 系統的成本 = API 調用費 + 向量庫費用 + 基礎設施

```
單次查詢成本計算：
1. Embedding 調用：1次 × $0.02/M tokens
2. LLM 調用：1次 × [輸入 $0.0005/K + 輸出 $0.0015/K]
3. VectorDB：按月固定費 ($0-100)

實例：100K 月查詢量
- Embedding：100K queries × 100 tokens × $0.02/M = $200
- LLM input (2K token avg)：100K × 2K × $0.0005/K = $100
- LLM output (500 token avg)：100K × 500 × $0.0015/K = $75
- Vector DB (Pinecone Free)：$0
= 月度總成本：~$375

同樣查詢用微調方案：
- GPU 租賃成本（A100）：$4000/月
- 標註成本：$50,000（一次性）
= 月度成本：$4000 > RAG 的 $375
= 微調需要 10+ 倍月查詢量才划算！
```

**10 大成本優化技巧**

| 優化手段 | 效果 | 難度 | ROI |
|--------|------|------|-----|
| 減少檢索 k 值（3→2） | -30% | ⭐ | 神 |
| 使用 Re-ranking | -20-40% vía 精準度 | ⭐⭐ | 超高 |
| 用本地 embedding 替代 API | -100% embedding 費 | ⭐⭐⭐ | 高 |
| Query 重寫減少不必要查詢 | -10-15% QPS | ⭐ | 高 |
| 批量查詢而非單個 | -40% | ⭐ | 高 |
| 緩存常見查詢結果 | -50% 熱查詢 | ⭐⭐ | 高 |
| 量化向量（8-bit） | -75% 存儲 | ⭐⭐⭐ | 中 |
| 分級 LLM（簡單→3.5，複雜→4） | -30-40% | ⭐⭐ | 高 |
| 使用 Cheaper LLM（Mistral vs GPT） | -70-80% | ⭐ | 超高 |
| Function Calling 替代 Non-Deterministic | -40% 重試 | ⭐⭐⭐ | 中 |

**實作範例：全能優化方案**

```python
from langchain.cache import RedisCache
from langchain.callbacks.tracers import LangChainTracer
import redis

# 設定快取（避免重複查詢）
redis_client = redis.Redis()
langchain.llm_cache = RedisCache(redis_client=redis_client)

# 方案 1：簡單查詢用便宜模型
def route_to_llm(question):
    # 簡單問題特徵：短、單一主題、常見詞
    if len(question) < 20 and question.count(' ') < 5:
        return "gpt-3.5-turbo"  # 便宜 3 倍
    else:
        return "gpt-4"  # 複雜問題用好模型

# 方案 2：本地 embedding + Pinecone 混合
from langchain_community.embeddings import HuggingFaceEmbeddings

local_embeddings = HuggingFaceEmbeddings(
    model_name="multilingual-e5-large"  # 本地，0 成本
)

# Query 高峰期用本地，低峰期才用 API embedding
if is_peak_hours():
    embeddings = local_embeddings  # 省 $200/月
else:
    embeddings = OpenAIEmbeddings()

# 方案 3：結構化輸出減少 Token
from langchain.output_parsers import StructuredOutputParser

parser = StructuredOutputParser.from_response_schemas([
    ResponseSchema(name="answer", description="直接答案（50 字以內）"),
    ResponseSchema(name="sources", description="來源文檔名錄"),
])

# 強制格式化限制 LLM 輸出大小 → 省 30-50% Token
```

---

### 4.3 故障診斷完全指南

**症狀 1：檢索結果差（Recall < 60%）**

診斷樹：
```
檢索差？
├─ 查詢不清楚？
│  └─ 修復：加 Query 改寫 (LLM 改寫查詢)
│
├─ 知識庫範圍不足？
│  └─ 修復：擴展知識庫 + 重建索引
│
├─ Embedding 模型不適配？
│  └─ 修復：試試 domain-specific 模型
│        (法律：法律-bert，科技：code-bert)
│
└─ 檢索參數差？
   ├─ k 值太小（2-3 太少）→ 試 k=5-10
   ├─ 相似度門檻太高 → 降低 threshold
   └─ 混合檢索比例不對 → vector_weight 試 0.3-0.7
```

**症狀 2：答案含幻覺（Hallucination Rate > 15%）**

診斷樹：
```
幻覺多？
├─ LLM 溫度過高？
│  └─ 修復：temperature 0.7 → 0.3
│
├─ Prompt 太寬鬆？
│  └─ 修復：加入「如果不知道，回答『我不知道』」限制
│
├─ 檢索到的文檔質量差？
│  └─ 修復：加 Re-ranking 過濾
│
└─ 文檔本身有誤？
   └─ 修復：檢查知識庫文檔準確性
```

**症狀 3：系統變慢（Latency > 5s）**

診斷樹：
```
慢？
├─ Embedding 調用慢？
│  ├─ 用批量 API（100 queries 一次）
│  └─ 改用本地 embedding（Sentence Transformer）
│
├─ 向量搜索慢？
│  ├─ 檢查向量庫索引類型（FAISS: Flat → IVF，質量 -2% 速度 +100x）
│  └─ 用 GPU 加速（faiss-gpu）
│
├─ LLM 推理慢？
│  ├─ 減少檢索文檔數（5→2）
│  ├─ 用流式輸出（邊生成邊發送，用戶體感快）
│  └─ 換便宜快模型（GPT-3.5 vs GPT-4）
│
└─ 序列化/數據傳輸慢？
   └─ 加快存儲層（Redis 快 100x vs 磁盤）
```

**完整診斷流程（15 分鐘診斷）**

```python
def diagnose_rag_system(
    sample_questions: List[str] = None,
    sample_size: int = 10
):
    """快速診斷 RAG 系統問題"""
    
    if sample_questions is None:
        sample_questions = [
            "什麼是 RAG？",
            "如何選擇 chunk_size？",
            "有什麼向量庫推薦？",
            "成本怎麼計算？",
            "怎樣降低幻覺率？"
        ]
    
    print("=" * 50)
    print("🔍 RAG 系統診斷開始")
    print("=" * 50)
    
    # 指標收集
    retrieval_times = []
    llm_times = []
    accuracy_scores = []
    
    for question in sample_questions[:sample_size]:
        print(f"\n▶ 測試：{question}")
        
        # 1. 檢索性能
        t1 = time.time()
        docs = retriever.get_relevant_documents(question, k=5)
        retrieval_time = time.time() - t1
        retrieval_times.append(retrieval_time)
        print(f"  ├─ 檢索時間：{retrieval_time:.2f}s")
        print(f"  ├─ 找到文檔：{len(docs)}")
        
        # 2. LLM 生成性能
        t2 = time.time()
        response = llm_chain.invoke({"context": docs, "question": question})
        llm_time = time.time() - t2
        llm_times.append(llm_time)
        print(f"  └─ LLM 時間：{llm_time:.2f}s")
        
    # 分析指標
    print("\n" + "=" * 50)
    print("📊 診斷結果")
    print("=" * 50)
    
    avg_retrieval = sum(retrieval_times) / len(retrieval_times)
    avg_llm = sum(llm_times) / len(llm_times)
    total_time = avg_retrieval + avg_llm
    
    print(f"\n【檢索性能】")
    print(f"  平均檢索時間：{avg_retrieval:.3f}s")
    if avg_retrieval > 1.0:
        print(f"  ⚠️  檢索太慢！建議：")
        print(f"     - 使用 GPU 加速（faiss-gpu）")
        print(f"     - 減少搜索 k 值（5→2）")
    
    print(f"\n【LLM 性能】")
    print(f"  平均 LLM 時間：{avg_llm:.3f}s")
    if avg_llm > 2.0:
        print(f"  ⚠️  LLM 推理太慢！建議：")
        print(f"     - 用流式輸出")
        print(f"     - 換更快的模型（GPT-3.5）")
    
    print(f"\n【總體延遲】")
    print(f"  總時間：{total_time:.3f}s")
    if total_time > 3.0:
        print(f"  ❌ 不符合用戶期望（< 3s），需優化")
    else:
        print(f"  ✅ 符合性能要求")
```

---

### 4.4 生產環境部署檢查清單

**Pre-Deployment 驗收標準**

```markdown
## 🚀 生產上線前 15 點檢查

#### 功能性檢查
- [ ] Recall@5 ≥ 80%（檢索滿足率）
- [ ] 幻覺率 < 10%（誤導信息）
- [ ] 系統應答時间 < 3 秒（用戶期望）
- [ ] 多語言支持驗證（中/英/其他）
- [ ] 特殊字符/emoji 處理正常
- [ ] 敏感信息過濾機制已驗收

#### 穩定性檢查
- [ ] 過載測試：100 QPS 下無崩潰
- [ ] 故障恢復：服務中斷 1 小時自動恢復
- [ ] 向量索引定期備份（每天 1 次）
- [ ] LLM API 故障降級方案（備用模型）
- [ ] 知識庫更新不中斷服務

#### 安全性檢查
- [ ] API Key 使用環境變數（不硬編碼）
- [ ] 用戶查詢日誌加密存儲
- [ ] PII（個人信息）自動脫敏
- [ ] Rate Limit 防止濫用（如 100 req/min/user）
- [ ] 輸入檢查防止 Prompt Injection

#### 可觀測性檢查
- [ ] 日誌系統記錄所有查詢 + 回答
- [ ] 性能指標：Latency、Recall、成本 dashboard
- [ ] 異常告警：Recall 突降、成本突升
- [ ] A/B 測試框架已部署

#### 文檔與培訓
- [ ] 系統架構文檔完成
- [ ] API 文檔已發佈
- [ ] 故障排除指南已編寫
- [ ] 操作人員已培訓
- [ ] 非技術人員可用（Web UI）
```

**上線後 1-4 周監控指標**

```python
# Weekly KPI Dashboard
kpis = {
    "week_1": {
        "total_queries": 10000,
        "avg_latency": 1.8,  # 秒
        "recall_rate": 0.82,  # 82%
        "hallucination_rate": 0.07,  # 7%
        "api_cost": "$87",
        "user_satisfaction": 4.2  # 1-5 分
    },
    "week_2": {...}
}

# 告警規則
ALERT_RULES = {
    "latency_spike": ("avg_latency", ">", 5),  # 超過 5 秒
    "recall_drop": ("recall_rate", "<", 0.70),  # 降到 70% 以下
    "cost_surge": ("api_cost", ">", 500),  # 月成本超 $500
    "satisfaction_drop": ("user_satisfaction", "<", 3.5),  # 評分下降
}
```

---

## 🎯 進階實驗計畫（如有時間）

若完成第一、二階段還有餘裕，可嘗試：

1. **對標 Production RAG**：
   - [ ] 實施 streaming 回答（邊生成邊推送給前端）
   - [ ] 加入對話記憶（multi-turn conversation）
   - [ ] 實施用戶反饋迴圈（標記好評/差評，用於微調）

2. **成本優化**：
   - [ ] 用 SentenceTransformers + ONNX 加速嵌入 (300x faster)
   - [ ] 實施向量量化（8-bit）降低存儲
   - [ ] 混合使用 gpt-3.5 (簡單) 和 gpt-4 (複雜) 節省成本

3. **評估進階**：
   - [ ] 自動化評估管道（CI/CD 集成）
   - [ ] 對比多套系統配置 (A/B test 數據驅動)
   - [ ] 建立 dashboard 實時監控 Recall/幻覺率

4. **應用場景拓展**：
   - [ ] 多語言 RAG（中文、英文、日文）
   - [ ] 長文本摘要（100 頁 PDF → single QA pair）
   - [ ] 多模態 RAG（圖片 + 文字）

---
