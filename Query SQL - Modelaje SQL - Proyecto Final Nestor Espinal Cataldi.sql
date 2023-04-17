---- QUERY
SELECT Modelos_Coche.Modelo_coche, ID_Marcas.NombreMarcas, id_grupo.Nombre_grupo, coches_de_la_empresa.Fecha_de_compra, coches_de_la_empresa.Matricula, Color_del_Coche.Color_del_Coche, coches_de_la_empresa.kilometraje, compania_aseguradora.IDCompanias 
FROM proyectofinal_NestorEspinal.coches_de_la_empresa 
JOIN proyectofinal_NestorEspinal.Modelos_Coche ON coches_de_la_empresa.Modelo_coche = Modelos_Coche.Modelo_coche 
JOIN proyectofinal_NestorEspinal.ID_Marcas ON Modelos_Coche.ID_marcas = ID_Marcas.ID_marcas 
JOIN proyectofinal_NestorEspinal.id_grupo ON ID_Marcas.ID_marcas = id_grupo.ID_Marcas 
JOIN proyectofinal_NestorEspinal.Color_del_Coche ON coches_de_la_empresa.Color_del_coche = Color_del_Coche.Color_del_Coche 
JOIN proyectofinal_NestorEspinal.compania_aseguradora ON coches_de_la_empresa.insurance_company = compania_aseguradora.insurance_company;