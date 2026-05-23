# Practical Elevator Traffic Review Tool v4

## Updates

- Traffic result now shows PASS / FAIL.
- Final recommendation shows how to solve failed traffic for each lift bank and scenario.
- Recommendation checks:
  - number of lifts
  - car capacity
  - speed
  - control system
  - zoning / sectoring
- Passenger, service and fireman lift pit/overhead remain separated.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

If Streamlit is not recognized:

```bash
python -m streamlit run app.py
```
