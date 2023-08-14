# Analyse des questions Stack Overflow liées à "GitHub Actions"

Dans cette analyse, nous allons compter le nombre total de questions posées sur Stack Overflow entre le 1er juillet 2018 et le 30 juin 2022, qui sont liées à "GitHub Actions". Pour ce faire, nous allons effectuer une requête SQL sur la base de données publique de BigQuery de Stack Overflow.

## Requête SQL

La requête SQL ci-dessous sélectionne le nombre total de questions qui satisfont les critères spécifiés :

Elles ont été créées entre le 1er juillet 2018 et le 30 juin 2022.
Le titre ou le corps de la question contient le terme "GitHub Actions" (en ignorant la casse).
Elles sont taguées avec "github-actions".

```sql
SELECT count(*)
FROM `bigquery-public-data.stackoverflow.posts_questions`
WHERE 
    creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
    AND (
        LOWER(title) LIKE '%github actions%' 
        OR LOWER(body) LIKE '%github actions%' 
        
        OR id IN (
            SELECT id
            FROM (
                SELECT id, SPLIT(tags, '|') tags
                FROM `bigquery-public-data.stackoverflow.posts_questions`
                WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
            ) 
            CROSS JOIN UNNEST(tags) flattened_tags
            WHERE LOWER(flattened_tags) = 'github-actions'
        )
    ); ```
    

### Résultat

Le résultat de la requête SQL est le nombre total de questions qui répondent à ces critères, dans notre cas, **6149** questions.

### Conclusion

Grâce à cette analyse, nous avons pu compter le nombre de questions liées à "GitHub Actions" posées sur Stack Overflow au cours de la période spécifiée. 


# Analyse des Mentions de "GitHub Actions" dans les Historiques des Posts

Dans cette analyse, nous comptons le nombre total d'occurrences où l'expression "GitHub Actions" a été mentionnée dans les historiques des posts de Stack Overflow entre le 1er juillet 2018 et le 30 juin 2022.

## Requête SQL

Pour obtenir le nombre d'occurrences où "GitHub Actions" a été mentionné dans les historiques des posts, la requête SQL suivante peut être utilisée :

```sql
SELECT COUNT(post_id)
FROM `bigquery-public-data.stackoverflow.post_history`
WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
AND (LOWER(text) LIKE '%github actions%').


### Résultat

Après avoir exécuté la requête SQL, nous obtenons le nombre total d'occurrences où "GitHub Actions" a été mentionné, soit **9897** occurrences.

#### Conclusion

Cette analyse révèle que l'expression "GitHub Actions" a été mentionnée **9897** fois dans les historiques des posts de Stack Overflow entre juillet 2018 et juin 2022. Cela suggère une utilisation significative de GitHub Actions dans les discussions et les interactions des développeurs sur la plateforme.



 
# Analyse des Questions avec Réponses Acceptées

Dans cette analyse, nous allons compter le nombre total de questions ayant des réponses acceptées sur Stack Overflow entre le 1er juillet 2018 et le 30 juin 2022.

### Requête SQL

Pour obtenir le nombre de questions avec des réponses acceptées, la requête SQL suivante peut être utilisée :

```sql
SELECT COUNT(*)
FROM `vital-program-390504.nath.final`
WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
  AND accepted_answer_id IS NOT NULL;


### Résultat

Après avoir exécuté la requête SQL, nous obtenons le nombre total de **2389** questions avec des réponses acceptées.

### Conclusion

Cette analyse nous permet de déterminer le nombre de questions sur Stack Overflow ayant des réponses acceptées entre juillet 2018 et juin 2022. Cela peut nous fournir des informations importantes sur l'engagement de la communauté envers la résolution de problèmes et la qualité des réponses fournies.


## Analyse des Questions sans Réponses Acceptées

Dans cette analyse, nous allons compter le nombre total de questions n'ayant pas de réponse acceptée sur Stack Overflow entre le 1er juillet 2018 et le 30 juin 2022.

### Requête SQL

Pour obtenir le nombre de questions sans réponse acceptée, la requête SQL suivante peut être utilisée :

```sql
SELECT COUNT(*)
FROM `vital-program-390504.nath.final`
WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
  AND accepted_answer_id IS NULL;

### Résultat

Après avoir exécuté la requête SQL, nous obtenons le nombre total de **3760** questions sans réponse acceptée.

### Conclusion

Cette analyse nous permet de déterminer le nombre de questions sur Stack Overflow qui n'ont pas de réponse acceptée entre juillet 2018 et juin 2022. Cela peut nous fournir des informations intéressantes sur les domaines où la communauté peut avoir besoin d'une assistance supplémentaire ou où les réponses peuvent ne pas être considérées comme résolues.



## Analyse des Questions sans Réponses

Dans cette analyse, nous allons compter le nombre total de questions qui n'ont aucune réponse sur Stack Overflow entre le 1er juillet 2018 et le 30 juin 2022.

### Requête SQL

Pour obtenir le nombre de questions sans réponses, la requête SQL suivante peut être utilisée :

```sql
SELECT COUNT(*)
FROM `vital-program-390504.nath.final`
WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
  AND answer_count = 0 . 


  #### Résultat

Après avoir exécuté la requête SQL, nous obtenons le nombre total de **1725** questions sans réponse.

### Conclusion

Cette analyse nous permet de déterminer le nombre de questions sur Stack Overflow qui n'ont pas de réponse entre juillet 2018 et juin 2022. Cela peut nous fournir des informations sur les sujets pour lesquels les réponses sont moins fréquentes ou peut-être plus difficiles à obtenir.


# Analyse des Questions avec Réponses

Dans cette analyse, nous allons compter le nombre total de questions qui ont au moins une réponse sur Stack Overflow entre le 1er juillet 2018 et le 30 juin 2022.

### Requête SQL

Pour obtenir le nombre de questions avec au moins une réponse, la requête SQL suivante peut être utilisée :

```sql
SELECT COUNT(*)
FROM `vital-program-390504.nath.final`
WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
  AND answer_count > 0;

  ### Résultat

Après avoir exécuté la requête SQL, nous obtenons le nombre total de **4424** questions avec au moins une réponse.

### Conclusion

Cette analyse nous permet de déterminer le nombre de questions sur Stack Overflow qui ont au moins une réponse entre juillet 2018 et juin 2022. Cela peut nous fournir des informations sur les sujets qui suscitent des discussions actives et des interactions au sein de la communauté des développeurs.


## Analyse des Utilisateurs Actifs

Dans cette analyse, nous allons extraire les informations sur les utilisateurs actifs de Stack Overflow entre le 1er juillet 2018 et le 30 juin 2022. Un utilisateur est considéré comme actif s'il a posé une question ou fourni une réponse pendant cette période.

### Requête SQL

Pour obtenir les informations sur les utilisateurs actifs, la requête SQL suivante peut être utilisée :

```sql
SELECT DISTINCT
  u.id AS user_id,
  u.display_name AS user_name
FROM (
  SELECT id AS post_id, owner_user_id
  FROM `vital-program-390504.nath.final`
  WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
  
) AS q
LEFT JOIN (
  SELECT parent_id, owner_user_id
  FROM `bigquery-public-data.stackoverflow.posts_answers`
  WHERE creation_date

 ### Résultat

Après avoir exécuté la requête SQL, nous obtenons le nombre total de **6986** des utilisateurs distincts.

### Conclusion

Cette analyse nous permet d'identifier les utilisateurs actifs de Stack Overflow pendant la période de juillet 2018 à juin 2022. En combinant les informations sur les questions et les réponses, nous pouvons avoir un aperçu des membres actifs de la communauté qui contribuent activement à la plateforme.

# Analyse des Tags Stack Overflow

Dans cette analyse, nous allons compter et calculer la proportion des tags les plus courants sur Stack Overflow entre le 1er juillet 2018 et le 30 juin 2022".

### Requête SQL

Pour obtenir les informations sur les tags les plus courants, la requête SQL suivante peut être utilisée :

```sql
WITH TagCounts AS (
  SELECT tag, COUNT(*) AS tag_count
  FROM (
    SELECT SPLIT(tags, '|') AS tags
    FROM `vital-program-390504.nath.final`
    WHERE  creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
  )
  CROSS JOIN UNNEST(tags) AS tag
  WHERE tag != 'github-actions'
  GROUP BY tag
)

SELECT
  'github-action' AS tag,
  COUNT(*) AS tag_count,
  (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()) AS percentage
FROM `vital-program-390504.nath.final`
WHERE  creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
UNION ALL
SELECT
  tag,
  tag_count,
  (tag_count * 100.0 / SUM(tag_count) OVER ()) AS percentage
FROM TagCounts
ORDER BY tag_count DESC;

### Résultat

Après avoir exécuté le script SQL, vous obtiendrez une liste des tags les plus courants avec le nombre de mentions (tag_count) ainsi que le pourcentage.

### Conclusion

Cette analyse nous permet de déterminer les tags les plus courants sur Stack Overflow entre juillet 2018 et juin 2022. Cela peut nous fournir des informations sur les sujets les plus discutés au sein de la communauté des développeurs.


# Analyse des Tags Moyens par Question

Dans cette analyse, nous allons calculer le nombre moyen de tags par question sur Stack Overflow entre le 1er juillet 2018 et le 30 juin 2022.

### Requête SQL

Pour calculer le nombre moyen de tags par question, la requête SQL suivante peut être utilisée :

```sql
WITH TagCounts AS (
  SELECT id, ARRAY_LENGTH(SPLIT(tags, '|')) AS tag_count
  FROM `vital-program-390504.nath.final`
  WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
)

SELECT
  'Average Tags per Question' AS tag,
  ROUND(AVG(tag_count), 2) AS average_tags_per_question
FROM TagCounts;

### Résultat

Après avoir exécuté la requête SQL, vous obtiendrez le nombre moyen de tags par question (3.18), arrondi à deux décimales.

### Conclusion

Cette analyse nous permet de calculer le nombre moyen de tags par question sur Stack Overflow pendant la période de juillet 2018 à juin 2022. Cela peut nous fournir des informations sur la façon dont les questions sont étiquetées et catégorisées par la communauté des développeurs.



# Analyse des Réponses Moyennes par Question

Dans cette analyse, nous allons calculer le nombre moyen de réponses par question sur Stack Overflow entre le 1er juillet 2018 et le 30 juin 2022.

### Requête SQL

Pour calculer le nombre moyen de réponses par question, la requête SQL suivante peut être utilisée :

```sql
SELECT
  'Average Answers per Question' AS metric,
  ROUND(AVG(answer_count), 2) AS average_answers_per_question
FROM (
  SELECT parent_id, COUNT(*) AS answer_count
  FROM `bigquery-public-data.stackoverflow.posts_answers`
  WHERE parent_id IN (
    SELECT id
    FROM `vital-program-390504.nath.final`
    WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
  )
  GROUP BY parent_id
);

### Résultat

Après avoir exécuté la requête SQL, vous obtiendrez le nombre moyen de réponses par question (1.38), arrondi à deux décimales.

#### Conclusion

Cette analyse nous permet de calculer le nombre moyen de réponses par question sur Stack Overflow pendant la période de juillet 2018 à juin 2022. Cela peut nous fournir des informations sur la fréquence des interactions et des discussions autour des questions posées par la communauté des développeurs.
