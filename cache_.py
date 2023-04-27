class Cache:
    def __init__(self) -> None:
        self._cache: dict = {}

        # Cache sample:
        # {
        #   id: {
        #       status: "processing" | "complete",
        #       result: {}
        #   }
        # }

    def get_all(self) -> dict:
        return self._cache

    def get(self, uuid: str) -> bool | KeyError:
        if uuid not in self._cache:
            return KeyError(f"{uuid} not found in cache")
        return self._cache[uuid]

    def add(self, uuid: str) -> dict:
        self._cache[uuid] = {
            "status": "processing",
            "result": {},
        }
        return self._cache[uuid]

    def update(self, uuid: str, status: str, result: dict):
        self._cache[uuid]["status"] = status
        self._cache[uuid]["result"] = result
        return self._cache[uuid]
