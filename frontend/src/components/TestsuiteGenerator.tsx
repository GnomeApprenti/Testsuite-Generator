import React, { useState } from 'react';

interface ApiResponse {
  testsuite: string;
}

interface CoverageResponse {
  coverage_report: string;
}

export const TestsuiteGenerator: React.FC = () => {
  const [projectPath, setProjectPath] = useState('');
  const [testsPath, setTestsPath] = useState('');
  const [sourcePath, setSourcePath] = useState('');
  const [testsuite, setTestsuite] = useState<string | null>(null);
  const [coverage, setCoverage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    setTestsuite(null);
    try {
      const response = await fetch('http://localhost:8080/generate_testsuite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: projectPath }),
      });
      if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
      const data: ApiResponse = await response.json();
      setTestsuite(data.testsuite);
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  const handleCalculateCoverage = async () => {
    setLoading(true);
    setError(null);
    setCoverage(null);
    try {
      const response = await fetch('http://localhost:8080/calculate_coverage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tests_path: testsPath, source_path: sourcePath }),
      });
      if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
      const data: CoverageResponse = await response.json();
      setCoverage(data.coverage_report);
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <style>{`
        input, button {
          font-size: 16px;
        }
        pre {
          white-space: pre-wrap;
          word-wrap: break-word;
        }
      `}</style>

      <h1 style={styles.heading}>🧪 Testsuite Generator</h1>

      <div style={styles.card}>
        <h2 style={styles.subheading}>Generate Testsuite</h2>
        <input
          type="text"
          placeholder="Enter project path"
          value={projectPath}
          onChange={(e) => setProjectPath(e.target.value)}
          style={styles.input}
        />
        <button onClick={handleGenerate} style={styles.buttonPrimary}>
          Generate
        </button>
      </div>

      <div style={styles.card}>
        <h2 style={styles.subheading}>📊 Calculate Coverage</h2>
        <input
          type="text"
          placeholder="Path to tests"
          value={testsPath}
          onChange={(e) => setTestsPath(e.target.value)}
          style={styles.input}
        />
        <input
          type="text"
          placeholder="Path to source"
          value={sourcePath}
          onChange={(e) => setSourcePath(e.target.value)}
          style={styles.input}
        />
        <button onClick={handleCalculateCoverage} style={styles.buttonSuccess}>
          Calculate Coverage
        </button>
      </div>

      {loading && <p style={styles.loading}>⏳ Loading...</p>}
      {error && <p style={styles.error}>❌ Error: {error}</p>}

      {testsuite && (
        <div style={{ ...styles.resultCard, backgroundColor: '#f8f9fa' }}>
          <h2 style={styles.resultTitle}>🧾 Generated Testsuite:</h2>
          <pre>{testsuite}</pre>
        </div>
      )}

      {coverage && (
        <div style={{ ...styles.resultCard, backgroundColor: '#e6ffed' }}>
          <h2 style={{ ...styles.resultTitle, color: '#2f855a' }}>✅ Coverage Report:</h2>
          <pre>{coverage}</pre>
        </div>
      )}
    </div>
  );
};

// Basic CSS-in-JS style object
const styles: { [key: string]: React.CSSProperties } = {
  container: {
    maxWidth: '800px',
    margin: '0 auto',
    padding: '2rem',
    fontFamily: 'Arial, sans-serif',
    color: '#333',
  },
  heading: {
    fontSize: '2rem',
    marginBottom: '1rem',
  },
  subheading: {
    fontSize: '1.2rem',
    marginBottom: '0.5rem',
  },
  card: {
    background: '#fff',
    borderRadius: '10px',
    boxShadow: '0 4px 10px rgba(0,0,0,0.1)',
    padding: '1.5rem',
    marginBottom: '2rem',
  },
  input: {
    display: 'block',
    width: '100%',
    padding: '0.75rem',
    marginBottom: '1rem',
    borderRadius: '6px',
    border: '1px solid #ccc',
    boxSizing: 'border-box',
  },
  buttonPrimary: {
    backgroundColor: '#007bff',
    color: '#fff',
    padding: '0.6rem 1.2rem',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  buttonSuccess: {
    backgroundColor: '#28a745',
    color: '#fff',
    padding: '0.6rem 1.2rem',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  loading: {
    color: '#555',
    fontStyle: 'italic',
  },
  error: {
    color: 'red',
  },
  resultCard: {
    borderRadius: '10px',
    padding: '1rem',
    marginTop: '1rem',
  },
  resultTitle: {
    fontSize: '1.1rem',
    marginBottom: '0.5rem',
  },
};

