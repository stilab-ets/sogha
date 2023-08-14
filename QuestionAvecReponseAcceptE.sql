SELECT count(*)
FROM `vital-program-390504.nath.final`
WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
  AND accepted_answer_id is not null
  