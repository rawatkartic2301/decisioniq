# Automated Timesheet Anomaly and Fraud Auditing Rules (2024)
*Effective: January 1, 2024 | Internal Audit & Machine Learning Analytics*

## 1. Anomaly Flagging Criteria
The internal audit engine automatically flags timesheets meeting any of the following statistical conditions:
1. **Impossible Daily Hours**: Any daily entry exceeding 16.0 hours logged in a single day is marked for mandatory manager review. Entries above 18.0 hours are automatically placed on hold.
2. **Duplicate Project Log**: Multiple entries submitted by the same employee for the same project code on the same date.
3. **Weekend Spikes**: An employee with zero weekend work history who suddenly logs >10 hours on a Saturday or Sunday without on-call assignment.
4. **Statistical Outlier**: Hours logged exceeding three standard deviations (z-score > 3.0) from the departmental monthly average.

## 2. Investigation Protocol
Flagged entries must be verified with Jira/GitHub sprint activity logs within five business days.
