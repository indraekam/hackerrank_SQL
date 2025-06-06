SELECT C.company_code, C.founder, 
  COUNT(DISTINCT E.LEAD_MANAGER_CODE) AS COUNT_LM,
  COUNT(DISTINCT E.SENIOR_MANAGER_CODE) AS COUNT_SM,
  COUNT(DISTINCT E.MANAGER_CODE) AS COUNT_M,
  COUNT(DISTINCT E.EMPLOYEE_CODE) AS COUNT_E
FROM COMPANY AS C
INNER JOIN EMPLOYEE AS E
ON C.COMPANY_CODE =  E.COMPANY_CODE
GROUP BY C.COMPANY_CODE, C.FOUNDER
ORDER BY C.COMPANY_CODE;
