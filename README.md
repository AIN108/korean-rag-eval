# korean-rag-eval

로컬 LLM(Ollama)만으로 한국어 문서 RAG의 구성 요소 — **청크 크기, 임베딩 모델, reranker, LLM** — 를 바꿔 가며 검색 정확도와 답변 정확도를 같은 조건에서 비교하는 실험 도구입니다. API 비용 없이 소비자급 GPU 한 장에서 돌아가는 것을 목표로 합니다.

> 결과 요약: _(실험 후 여기에 대표 결과 표와 그림을 넣습니다)_

## 무엇을 측정하나

| 지표 | 의미 | 계산 방식 |
|---|---|---|
| Recall@k | 정답 근거가 상위 k개 검색 결과에 들어왔는가 | QA셋에 표시한 근거 문장 구간을 50% 이상 덮는 청크가 있으면 hit |
| MRR | 근거 청크가 몇 번째로 검색됐는가 | 1 / (첫 hit 순위) 평균 |
| 정답 포함률 | 답변에 정답 문자열이 들어 있는가 | 공백·문장부호 제거 후 포함 여부 (짧은 사실형 정답용 보조 지표) |
| 채점 정확도 | 답변이 정답과 의미상 일치하는가 | 별도 채점 LLM이 JSON으로 판정 (해석 불가 응답은 집계 제외) |
| 평균 지연 | 답변 생성 시간 | 질문당 LLM 호출 시간 |

검색 hit를 "같은 문서인가"가 아니라 **근거 문장의 원문 위치**로 판정하기 때문에, 청크 크기를 바꿔도 공정하게 비교됩니다.

## 빠른 시작

전제: Python 3.10+, [Ollama](https://ollama.com) 설치 및 실행 중.

```bash
git clone https://github.com/AIN108/korean-rag-eval && cd korean-rag-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # reranker까지 쓰려면 ".[dev,rerank]"

ollama pull bge-m3 && ollama pull qwen2.5:7b   # 설정 파일에 적은 모델을 받습니다

python -m rag_eval validate --config configs/default.yaml   # 문서·QA셋 검증
python -m rag_eval run --config configs/retrieval_only.yaml -v   # 검색 지표만 (빠름)
python -m rag_eval run --config configs/default.yaml -v          # 전체 비교
```

결과는 `results/<name>-<시각>/`에 저장됩니다.

- `summary.md`, `summary.csv`: 설정 조합별 지표
- `rows.jsonl`: 질문별 검색 결과·답변·채점 (오답 분석용)
- `config.yaml`: 실행에 쓴 설정 사본 (재현용)

임베딩은 `results/.cache/`에 캐시되어, LLM만 바꿔 재실행할 때는 다시 계산하지 않습니다.

## 데이터 형식

`data/docs/`에 `.txt` / `.md` 문서(UTF-8)를 넣고, `data/qa/*.jsonl`에 한 줄에 하나씩 질문을 작성합니다.

```json
{"id": "q001", "question": "GPU 서버는 언제까지 예약해야 하나요?", "answer": "사용 3일 전까지",
 "doc_id": "sample_lab_rules.md", "evidence": "공용 GPU 서버는 예약 시스템을 통해 사용 3일 전까지 예약해야 한다."}
```

- `doc_id`: `docs_dir` 기준 상대 경로
- `evidence`: 정답 근거가 되는 원문 문장. 공백·줄바꿈 차이는 무시하고 찾습니다. 본문에 없으면 `validate`가 행 번호와 함께 알려 줍니다.

동봉된 `sample_lab_rules.md`는 동작 확인용으로 직접 작성한 가상 문서입니다. 실험에는 재배포가 허용된 공개 문서(예: 공공누리 제1유형 표시 자료)를 쓰고, 문서마다 출처와 라이선스를 `data/docs/SOURCES.md`에 기록하세요.

## 구조

```
rag_eval/
  config.py     설정 로드·검증
  corpus.py     문서 로드, 문장 경계를 고려한 문자 단위 청크 분할 (원문 위치 보존)
  qa.py         QA셋 로드, 근거 위치 검증
  retrieval.py  코사인 벡터 검색, 임베딩 캐시, CrossEncoder reranker
  ollama.py     Ollama /api/embed, /api/chat 클라이언트
  prompts.py    답변·채점 프롬프트
  metrics.py    Recall@k, MRR, 정답 포함률, 집계
  runner.py     설정 조합 순회
  report.py     CSV·Markdown 출력
tests/          Ollama 없이 돌아가는 단위·통합 테스트 (가짜 임베더/LLM 사용)
```

벡터 DB 대신 NumPy 행렬 곱으로 검색합니다. 문서 수천 개 규모에서는 충분히 빠르고, 의존성이 적어 결과 재현이 쉽기 때문입니다.

## 한계

- 청크 크기는 토큰이 아니라 **문자 수** 기준입니다. 모델마다 토크나이저가 달라, 비교 기준을 통일하려고 문자 수를 썼습니다.
- LLM 채점은 채점 모델의 편향을 그대로 따릅니다. 답변을 생성한 모델과 다른 모델로 채점하고, 일부 표본은 직접 검수하는 것을 권합니다.
- `temperature=0`, 고정 `seed`를 쓰지만 Ollama 버전이나 하드웨어에 따라 출력이 완전히 같지 않을 수 있습니다.

## 라이선스

코드: MIT. 데이터는 각 원본 문서의 라이선스를 따릅니다.
