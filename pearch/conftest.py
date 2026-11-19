import pytest


class ContractTestHooks:
    @pytest.hookspec(firstresult=True)
    def pytest_pearch_client_options(self):
        """Keyword arguments for the AsyncPearchClient that the contract tests build."""


def pytest_addhooks(pluginmanager):
    pluginmanager.add_hookspecs(ContractTestHooks)


def pytest_configure(config):
    config.addinivalue_line("markers", "contract_health: one fast search, the cheapest proof that the API answers")
    config.addinivalue_line("markers", "contract_full: every public client method against a live API")
