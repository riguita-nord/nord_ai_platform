"""Run with: python3 -m unittest discover -s tests -v"""
import sqlite3
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"server"))
import knowledge_router as router

class RoutingTests(unittest.TestCase):
    def test_intent(self):
        self.assertEqual(router.route_hint("Escreve um email"),"local")
        self.assertEqual(router.route_hint("Pesquisa na internet as últimas notícias"),"web")
        self.assertEqual(router.route_hint("https://example.org/docs"),"link")
    def test_normalize(self):
        self.assertEqual(router.cache_key("Como funciona um processador?"),router.cache_key("Como funciona um processador?"))
        self.assertNotEqual(router.cache_key("Como funciona um processador?"),router.cache_key("Como funciona uma placa gráfica?"))
    def test_cache(self):
        with sqlite3.connect(":memory:") as db:
            router.init(db)
            self.assertIsNone(router.recall(db,"processador"))
            router.save(db,"processador","Fonte: teste de texto", [{"title":"Documentação","url":"https://example.org/"}])
            entry=router.recall(db,"processador")
            self.assertIsNotNone(entry)
            self.assertEqual(entry["sources"][0]["title"],"Documentação")
    def test_dynamic(self):
        self.assertTrue(router.dynamic("preço atual"))
        self.assertFalse(router.dynamic("como funciona um processador"))

if __name__=="__main__":
    unittest.main()
