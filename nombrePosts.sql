SELECT COUNT( post_id)
FROM `bigquery-public-data.stackoverflow.post_history`
WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
AND (LOWER(text) LIKE '%github actions%');
