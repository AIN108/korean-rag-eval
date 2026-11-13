# 실제 평가 데이터 준비

이 도구는 문서와 정답이 연결된 질문셋으로 RAG 설정을 비교합니다. 기존 로컬 RAG 앱에서 실제로 검색하던 PDF를 먼저 사용하는 것을 권합니다. PDF를 많이 모으는 것보다, 정답과 근거를 직접 검수한 질문을 만드는 작업이 우선입니다.

`data/examples/`의 2개 문서·12문항은 **형식 설명용 허구 자료**입니다. 기존 `data/docs/`와 `data/qa/sample.jsonl`의 6문항도 동작 확인용입니다. 두 예시를 실제 성능 평가 데이터로 집계하지 않습니다.

## 1. 원문과 출처 정리

실제 자료는 기본적으로 Git에서 제외되는 아래 경로에 보관합니다. 출처 목록은 검색 대상인 `docs/` 밖에 둡니다.

```text
data/private/
  docs/
    country_a_edition.pdf
    country_b_edition.pdf
  qa.jsonl
  sources.jsonl
```

출처 목록에는 `doc_id`, 자료명, 판본·발행연도, 원본 URL, 확보일, 원본 파일 SHA-256, 이용 범위, `source_group`을 기록합니다. `doc_id`는 `docs_dir` 기준 상대 경로이며 대소문자까지 일치해야 합니다.

기존 `lags_ablation.yaml`에 맞추려면 [미 의회도서관 Country Studies](https://www.loc.gov/collections/country-studies/about-this-collection/) PDF를 사용할 수 있습니다. 이 자료의 연도별 서술과 수치는 해당 판본의 역사적 사실을 묻는 데 사용하고 현재 상황으로 바꾸어 묻지 않습니다. 영어 문서에 한국어로 질문하면 **교차 언어 검색 평가**이므로 결과에도 그 범위를 명시합니다.

[컬렉션 권리 안내](https://www.loc.gov/collections/country-studies/about-this-collection/rights-and-access/)는 일부 사진에 별도 권리가 있음을 설명합니다. 텍스트 중심으로 사용하고 해당 항목의 안내를 기록합니다. 다른 출판물도 같은 이용 조건이라고 간주하지 않습니다.

## 2. 평가기가 읽는 텍스트 확인

PDF와 HF 지원이 필요하면 저장소 루트에서 선택 의존성을 설치합니다.

```bash
python -m pip install -e ".[dev,pdf]"
python -m rag_eval export-text --config configs/user_data.example.yaml
```

이 명령은 QA 파일이 아직 없어도 실행됩니다. 매번 새 `results/annotation-text-.../` 폴더를 만들고, 정규화된 텍스트와 `index.json`을 저장합니다. QA의 `doc_id`에는 내보낸 `.txt` 이름이 아니라 `index.json`의 **원래 문서 ID**를 씁니다.

여기서 복사한 문장을 `evidences[].text`에 넣습니다. 공백·줄바꿈 차이는 무시하지만, PDF에서 단어가 깨지거나 하이픈이 생기면 그대로 확인해야 합니다. 스캔 PDF는 현재 OCR하지 않고 경고 후 제외하므로 먼저 텍스트 추출 상태를 확인합니다. 페이지 번호는 현재 평가기의 근거 위치 정보에 포함되지 않습니다.

## 3. 질문 작성과 검수

1차 목표 예시는 100문항입니다. 먼저 20~30문항으로 전체 절차를 확인한 뒤 확장합니다. 아래 배분은 제안이며 통계적으로 충분한 표본 수를 보장하는 기준은 아닙니다.

| 유형 | 예시 목표 | 검수할 사항 |
| --- | ---: | --- |
| 단순 사실 `fact` | 30 | 대상·시점·조건이 정답과 같은가 |
| 수치·단위 `numeric` | 25 | 값·단위·기준연도가 빠지지 않았는가 |
| 비교 `comparison` | 15 | 비교 대상 양쪽의 근거가 모두 있는가 |
| 복수 근거 `multi_evidence` | 15 | 답변에 필요한 각 문장을 모두 지정했는가 |
| 답변 불가 `unanswerable` | 15 | 검색 대상 전체에 답이 없는지 확인했는가 |
| 합계 | 100 | 중복·말바꾸기 문항을 별도 성과로 부풀리지 않는가 |

질문·정답 초안을 LLM으로 만들더라도 원문 대조를 거친 항목만 평가셋에 넣습니다. 문서 밖의 일반 지식이나 최신 정보를 정답에 추가하지 않습니다. 평가 모델이 틀린 질문을 보고 정답을 수정하는 대신, 원문에 비추어 잘못된 라벨인지 먼저 확인합니다.

### 답변 가능한 질문

실제 동작하는 형식 예시는 `data/examples/qa.jsonl`에 있습니다. 아래도 가상 규정에서 가져온 형식 예시입니다.

```json
{"id":"demo-dev-04","question":"방문 연구자가 GPU 서버를 쓰려면 누구의 승인을 받고 언제까지 예약해야 하나요?","answer":"지도교수의 승인을 받고 사용 3일 전까지 예약해야 한다.","evidences":[{"doc_id":"fictional_lab.md","text":"방문 연구자는 지도교수의 승인을 받은 경우에만 장비를 사용할 수 있다."},{"doc_id":"fictional_lab.md","text":"공용 GPU 서버는 사용 3일 전까지 예약해야 한다."}],"question_type":"multi_evidence","split":"dev","source_group":"fictional-lab"}
```

- `evidences`는 답변에 **모두 필요한 근거**의 목록입니다. 서로 대체 가능한 근거를 필수 근거처럼 나열하지 않습니다.
- 같은 문장이 반복되면 해당 근거 항목에 `occurrence: 2`처럼 1부터 시작하는 등장 순서를 지정할 수 있습니다. 생략하면 첫 번째 등장 위치를 사용합니다.
- `answer_aliases`에는 허용할 표기 차이를 넣습니다. 문자열 포함률에만 적용하며, 의미상 정답을 보장하는 지표는 아닙니다.
- 기존 `doc_id` + `evidence` 단일 근거 형식도 계속 사용할 수 있습니다. 같은 행에서 새 `evidences`와 혼용하지 않습니다.

### 답변 불가 질문

```json
{"id":"demo-dev-06","question":"GPU 서버를 1시간 사용하는 요금은 얼마인가요?","answerable":false,"question_type":"unanswerable","split":"dev","source_group":"fictional-lab"}
```

`answer`와 근거는 생략합니다. 참조 답변은 `문서에서 찾을 수 없습니다`로 고정됩니다. 한 문서에 없더라도 다른 검색 대상 문서에 답이 있으면 답변 불가 문항으로 만들면 안 됩니다. 근거가 없는 문항은 검색 회수율·LLM 채점의 분모에서 제외하고 답변 거부 정확도로 따로 평가합니다.

## 4. 개발용과 최종 평가용 분리

`split`은 `dev` 또는 `test`를 사용합니다. 이전 형식은 `unspecified`로 읽힙니다. 예를 들어 개발용 60문항, 최종 평가용 40문항으로 나누되 **같은 문서·자료 계열의 질문은 같은 source_group으로 묶어 한쪽에만 배정**합니다. 같은 책의 장이나 비슷한 판본을 양쪽에 나누면 내용이 겹칠 수 있습니다.

명시한 `source_group`이 dev/test에 동시에 있으면 검증이 실패합니다. group을 생략한 자료의 중복이나 서로 다른 파일의 유사 내용까지 자동 탐지하지는 않으므로 사람이 확인해야 합니다.

원문 검색 대상에는 dev/test의 정답 근거 문서를 모두 둡니다. 분리하는 것은 **튜닝에 쓰는 질문**입니다. 최종 평가용 근거 문서를 검색 대상에서 빼면 답을 찾을 수 없게 됩니다.

## 5. 검증과 실행

```bash
python -m rag_eval validate --config configs/user_data.example.yaml
python -m rag_eval run --config configs/user_data.example.yaml -v
```

첫 설정은 `qa_split: dev`, LLM 없이 검색만 비교합니다. `summary_by_type.md`에서 질문 유형별 누락을 먼저 확인합니다. 검색 조건을 좁힌 다음 `llm_models`를 추가하고, 선택적으로 답변 모델과 다른 `judge_model`을 지정합니다. 설정을 확정한 뒤 별도 이름의 설정 파일에 `qa_split: test`를 두고 최종 평가합니다.

기본 `top_k`를 같게 두어도 청크 크기가 다르면 LLM에 들어가는 문맥 길이가 달라집니다. 현재 도구는 입력 토큰 예산을 맞추지 않으므로 청크 비교 결과와 이 제약을 함께 기록합니다.

## 6. 결과 기록

- `rows.jsonl`: 질문별 검색 근거 ID, 각 필수 근거의 hit 순위, 답변, 채점 원문·실패 상태. 요청 실패가 있어도 해당 행을 남깁니다.
- `summary.md/csv`: 전체 결과. 유효 채점 정확도와 함께 채점 시도 수·성공 수·유효율을 봅니다. 실패를 실제 오답으로 확정하지 않습니다.
- `summary_by_type.md/csv`: 같은 조건을 질문 유형별로 나눈 결과.
- `manifest.json`: 선택한 질문 수·split, 유효 설정, 정규화 문서·QA·코드 해시, Python/패키지 버전과 실행 상태. 모델 가중치 digest·Ollama 버전·하드웨어는 자동으로 기록하지 않으므로 별도 메모합니다.

임베딩 캐시는 모델 이름과 입력 텍스트로 구분합니다. 같은 모델 태그의 가중치를 교체했으면 `results/.cache/`의 해당 `.npy` 캐시를 비우고 다시 측정합니다. 검색 시간에는 임베딩·모델 적재가 포함되지 않습니다.

대표 결과를 공개할 때에는 실제로 실행한 결과만 `docs/`에 복사하고 데이터·split·질문 수·채점 유효율을 함께 씁니다. 허구 예제나 가짜 모델을 쓴 테스트 결과를 실제 RAG 성능으로 소개하지 않습니다. 현재는 중단된 실험을 자동 재개하지 않으며, 원인을 해결한 뒤 새 실행 폴더에서 다시 시작합니다.
