# korean-rag-eval

로컬 LLM(Ollama)만으로 한국어 질문 RAG의 구성 요소 — **청크 크기, 임베딩 모델(접두어 포함), 질의 확장, reranker, LLM** — 를 바꿔 가며 검색 정확도와 답변 정확도를 같은 조건에서 비교하는 실험 도구입니다. API 비용 없이 소비자급 GPU 한 장에서 돌아가는 것을 목표로 합니다.

직접 만들어 쓰던 로컬 RAG 앱([AIN108/lags2.py](https://github.com/AIN108/lags2.py))에서 버전마다 바꿔 온 설계 선택(청크 크기, e5 vs bge-m3, 한→영 질의 확장)이 실제로 효과가 있었는지 정량으로 확인하려고 만들었습니다.

> 결과 요약: _(실험 후 여기에 대표 결과 표와 그림을 넣습니다)_

## 무엇을 측정하나

| 지표 | 의미 | 계산 방식 |
|---|---|---|
| Recall@k (기존 이름) | 필요한 근거 중 하나 이상을 찾았는가 | 답변 가능한 질문별 hit 비율. 단일 근거에서는 기존 동작과 동일 |
| 필수 근거 회수율 / 전체 근거 hit@k | 복수 근거를 얼마나 찾았는가 | 질문별 근거 회수 비율의 평균 / 필요한 근거를 모두 찾은 질문 비율 |
| MRR | 근거 청크가 몇 번째로 검색됐는가 | 1 / (첫 hit 순위) 평균 |
| 정답 포함률 | 답변에 정답 문자열이 들어 있는가 | 공백·문장부호 제거 후 포함 여부 (짧은 사실형 정답용 보조 지표) |
| 유효 채점 정확도 / 채점 유효율 | 답변이 정답과 일치하며 채점이 얼마나 유효한가 | JSON 해석에 성공한 채점의 정확도와 성공 수/전체 시도 수를 함께 보고 |
| 답변 불가 정답률 | 근거가 없을 때 정해진 답변 거부를 하는가 | 답변 불가 질문을 검색 지표와 분리하여 집계 |
| 단계별 평균 지연 | 검색·재정렬·답변 생성 시간 | 임베딩·모델 적재 시간은 제외 |

검색 hit는 "같은 문서인가" 대신 **근거 문장의 원문 위치**로 판정합니다. 다만 같은 top-k라도 청크 크기에 따라 문맥 길이가 달라지며, 현재 입력 토큰 예산을 맞추지는 않습니다.

실제 데이터는 [데이터 준비 안내](docs/DATASET_GUIDE.md)에 따라 준비하세요. 기존 6문항은 동작 확인용이고, [새 형식의 12문항 예제](data/examples/qa.jsonl)도 허구 자료이므로 대표 성능으로 사용하지 않습니다.

## 빠른 시작

전제: Python 3.10+, [Ollama](https://ollama.com) 설치 및 실행 중.

```bash
git clone https://github.com/AIN108/korean-rag-eval && cd korean-rag-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # HF 임베딩·reranker는 ".[dev,hf]", PDF는 ".[pdf]"

ollama pull bge-m3 && ollama pull qwen2.5:7b   # 설정 파일에 적은 모델을 받습니다

python -m rag_eval validate --config configs/default.yaml   # 문서·QA셋 검증
python -m rag_eval validate --config configs/schema_demo.yaml  # 복수 근거·답변 불가·split 예제 검증
python -m rag_eval run --config configs/retrieval_only.yaml -v   # 검색 지표만 (빠름)
python -m rag_eval run --config configs/default.yaml -v          # 전체 비교
```

결과는 `results/<name>-<시각>/`에 저장됩니다.

- `summary.md`, `summary.csv`: 설정 조합별 지표
- `summary_by_type.md`, `summary_by_type.csv`: 질문 유형별 지표
- `rows.jsonl`: 질문별 검색 결과·답변·채점 (오답 분석용)
- `config.yaml`: 실행에 쓴 설정 사본 (재현용)
- `manifest.json`: 유효 설정·데이터/코드 해시·런타임 버전·실행 상태

채점 실패는 `judge_failed_n`, `judge_coverage`로 드러나며 `judge_acc`는 유효 채점만을 분모로 삼습니다. `judge_correct_rate`는 전체 채점 시도 중 정답으로 확인된 비율입니다. 둘을 함께 보아야 하며 미채점 항목을 실제 오답으로 확정하는 지표는 아닙니다. 요청 오류가 남은 실행은 결과를 저장하되 CLI가 종료 코드 1을 반환합니다.

기본 `judge_model`은 `null`입니다. 자동 채점하려면 답변 모델과 다른 모델을 직접 지정하고 일부 표본을 원문과 대조하세요. 답변 모델과 같은 모델을 지정하면 경고가 출력됩니다.

임베딩은 `results/.cache/`에 캐시되어, LLM만 바꿔 재실행할 때는 다시 계산하지 않습니다.

## 실험 설정

설정 파일의 리스트 항목들의 모든 조합을 실행합니다.

```yaml
embed_models:
  - bge-m3                                   # 문자열 = Ollama 임베딩 모델
  - { name: bge-m3-hf, model: BAAI/bge-m3, backend: hf }   # Hugging Face (pip install -e ".[hf]")
  - name: e5-large+prefix                    # 같은 모델의 접두어 유무도 별도 조건으로 비교
    model: intfloat/multilingual-e5-large
    backend: hf
    query_prefix: "query: "
    passage_prefix: "passage: "

query_expansions:                            # 검색 질의에만 적용, LLM에는 원래 질문 전달
  - null
  - data/glossary/military_ko_en.yaml        # {한국어 용어: 영어 대응어} 용어집
```

### 기존 앱 설계 검증 실험 (`configs/lags_ablation.yaml`)

| 비교 항목 | 조건 | 출처 |
|---|---|---|
| 청크 | 1000/200 vs 800/100 | `rag.py` → v5 이후 |
| 임베딩 | multilingual-e5-large vs bge-m3 | v5·v6 vs `lags3.py` |
| e5 접두어 | 없음(원본 앱) vs `query:`/`passage:` | e5 모델 권장 사용법 |
| 질의 확장 | 없음 vs 한→영 용어집 | `preprocess_question` |
| LLM | qwen2.5:7b vs gemma3n:e2b | `lags5.py` 지원 모델 |

원본 앱은 LangChain `RecursiveCharacterTextSplitter`를 쓰고, 이 도구는 문장 경계를 고려한 자체 분할기를 쓰므로 청크 경계가 완전히 같지는 않습니다.

## 데이터 형식

`data/docs/`에 `.txt` / `.md` / `.pdf` 문서를 넣고, `data/qa/*.jsonl`에 한 줄에 하나씩 질문을 작성합니다. PDF는 `pip install -e ".[pdf]"`가 필요하며, 추출한 텍스트는 `results/.cache/text/`에 캐시됩니다. 텍스트 층이 없는 스캔 PDF는 경고 후 건너뜁니다.

`lags_ablation` 실험용 문서는 미 의회도서관 Country Studies(Federal Research Division) PDF를 `data/country_studies/`에 넣어 사용합니다. 용량 때문에 저장소에는 포함하지 않습니다(`.gitignore`). 미국 연방정부 저작물이라 퍼블릭 도메인으로 알려져 있지만, 질의셋을 공개하기 전에 각 판본의 출처 페이지에서 이용 조건을 확인하세요. 상용 출판물(예: IISS The Military Balance)은 질의셋 근거 문장으로도 쓰지 않습니다.

```json
{"id": "q001", "question": "GPU 서버는 언제까지 예약해야 하나요?", "answer": "사용 3일 전까지",
 "doc_id": "sample_lab_rules.md", "evidence": "공용 GPU 서버는 예약 시스템을 통해 사용 3일 전까지 예약해야 한다."}
```

- `doc_id`: `docs_dir` 기준 상대 경로
- `evidence`: 정답 근거가 되는 원문 문장. 공백·줄바꿈 차이는 무시하고 찾습니다. 본문에 없으면 `validate`가 행 번호와 함께 알려 줍니다.

새 형식은 `evidences: [{doc_id, text, occurrence?}, ...]`로 필요한 근거 여러 개를 지정합니다. `answerable: false`인 질문은 정답·근거 없이 작성합니다. `answer_aliases`, `question_type`, `split`, `source_group`도 선택적으로 지원합니다. `qa_split: dev` 또는 `test`로 실행 대상을 선택하고, 같은 `source_group`이 두 split에 겹치면 검증이 실패합니다. 기존 단일 근거 형식은 계속 지원합니다.

동봉된 `sample_lab_rules.md`는 동작 확인용으로 직접 작성한 가상 문서입니다. 실제 자료의 출처·이용 범위는 검색 대상 밖인 `data/private/sources.jsonl` 등에 기록합니다. 실제 문서와 검수 중인 QA를 두는 `data/private/`는 Git에서 제외됩니다. PDF의 평가용 텍스트는 `python -m rag_eval export-text --config configs/user_data.example.yaml`로 내보낼 수 있습니다.

## 구조

```
rag_eval/
  config.py     설정 로드·검증
  corpus.py     문서(txt/md/pdf) 로드, 문장 경계를 고려한 문자 단위 청크 분할 (원문 위치 보존)
  qa.py         QA셋 로드, 근거 위치 검증
  expansion.py  용어집 기반 질의 확장
  retrieval.py  코사인 벡터 검색, 임베딩 캐시, HF 임베딩, CrossEncoder reranker
  ollama.py     Ollama /api/embed, /api/chat 클라이언트
  prompts.py    답변·채점 프롬프트
  provenance.py 데이터·설정·코드 지문과 실행 상태 기록
  metrics.py    Recall@k, MRR, 정답 포함률, 집계
  runner.py     설정 조합 순회
  report.py     CSV·Markdown 출력
tests/          Ollama 없이 돌아가는 단위·통합 테스트 (가짜 임베더/LLM 사용)
```

벡터 DB 대신 NumPy 행렬 곱으로 검색합니다. 문서 수천 개 규모에서는 충분히 빠르고, 의존성이 적어 결과 재현이 쉽기 때문입니다.

## 한계

- 청크 크기는 토큰이 아니라 **문자 수** 기준입니다. 모델마다 토크나이저가 달라, 비교 기준을 통일하려고 문자 수를 썼습니다.
- LLM 채점은 채점 모델의 편향을 그대로 따릅니다. 답변을 생성한 모델과 다른 모델로 채점하고, 일부 표본은 직접 검수하는 것을 권합니다.
- 정답 포함률은 부정문·조건 오류도 맞다고 잡을 수 있는 보조 지표입니다. 예를 들어 `48시간이 아니라 24시간`에도 `48시간`은 포함됩니다.
- 답변 불가 평가는 정해진 거부 문구와의 일치 여부를 측정하며, 모든 형태의 의미상 거부를 판정하지는 않습니다.
- 복수 근거의 각 hit는 개별 청크가 근거 구간을 설정 비율 이상 덮는지 판정합니다. 여러 청크의 부분 구간을 합치는 평가는 하지 않습니다.
- 중단된 실행의 행별 결과는 남지만 자동 재개는 지원하지 않습니다. 모델 태그를 유지한 채 가중치를 바꾸면 임베딩 캐시도 갱신해야 합니다.
- `temperature=0`, 고정 `seed`를 쓰지만 Ollama 버전이나 하드웨어에 따라 출력이 완전히 같지 않을 수 있습니다.

## 라이선스

코드: MIT. 데이터는 각 원본 문서의 라이선스를 따릅니다.
