SELECT CONCAT(NAME, '(', UPPER(LEFT(OCCUPATION, 1)), ')') AS FORMATNAME
FROM OCCUPATIONS
ORDER BY NAME;
SELECT CONCAT('There are a total of ', COUNT(OCCUPATION), ' ', LOWER(OCCUPATION), 's.') AS CONCLUCTION
FROM OCCUPATIONS
GROUP BY OCCUPATION
ORDER BY COUNT(OCCUPATION), LOWER(OCCUPATION);
