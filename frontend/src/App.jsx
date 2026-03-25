import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import Overview from './pages/Overview'
import Architecture from './pages/Architecture'
import RQ1 from './pages/RQ1'
import RQ2 from './pages/RQ2'
import RQ3 from './pages/RQ3'
import RQ4 from './pages/RQ4'
import RQ5 from './pages/RQ5'
import Findings from './pages/Findings'

export default function App() {
  return (
    <div className="flex h-screen bg-dash-bg overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto p-6">
          <Routes>
            <Route path="/"             element={<Overview />} />
            <Route path="/architecture" element={<Architecture />} />
            <Route path="/rq1"          element={<RQ1 />} />
            <Route path="/rq2"          element={<RQ2 />} />
            <Route path="/rq3"          element={<RQ3 />} />
            <Route path="/rq4"          element={<RQ4 />} />
            <Route path="/rq5"          element={<RQ5 />} />
            <Route path="/findings"     element={<Findings />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}
