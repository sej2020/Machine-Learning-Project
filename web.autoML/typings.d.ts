declare var process: Process;

interface Process {
    env: Env
}

interface Env {
    API_ROUTE: string
}

interface GlobalEnvironment {
    process: Process
}