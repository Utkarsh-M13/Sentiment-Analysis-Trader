import { useEffect, useState } from "react"
import SharpeGauge from "./SharpeGuage"
import { fetchSharpeRatio } from "../lib/fetchData";

const SharpeRatio = () => {
  const [sharpe, setSharpe] = useState(5); 

  useEffect(() => {
    async function fetchData() {
      try {
        const data = await fetchSharpeRatio();
        if (data === null || data === undefined) {
          console.warn("Received null or undefined Sharpe ratio, defaulting to 0");
          setSharpe(0);
          return;
        }
        setSharpe(data);
      } catch (err) {
        console.error("Failed to fetch Sharpe ratio:", err);
      }
    }

    fetchData();
  }, []);

  return (
    <div className="w-40 h-40 rounded-lg purple-shadow">
      <SharpeGauge value={sharpe}></SharpeGauge>
    </div>
  )
}

export default SharpeRatio