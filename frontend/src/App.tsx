import BnH from "./components/BnH"
import EquityCurve from "./components/EquityCurve"
import HeadlineLog from "./components/HeadlineLog"
import RiskExposure from "./components/RiskExposure"
import SharpeRatio from "./components/SharpeRatio"
import Topbar from "./components/Topbar"
import TotalEquity from "./components/TotalEquity"
import TradeLog from "./components/TradeLog"

function App() {

  return (
    <div className="w-full min-h-screen flex gap-6 flex-col">
      <Topbar></Topbar>
      <div className="h-75 w-full pl-7 pr-7 mt-30 flex gap-5">
        <EquityCurve></EquityCurve>
        <div className="h-75 w-40 flex flex-col gap-5">
          <TotalEquity></TotalEquity>
          <SharpeRatio></SharpeRatio>
        </div>
        <TradeLog></TradeLog>
      </div>
      <div className="h-75 w-full pl-7 pr-7 flex gap-5">
        <HeadlineLog></HeadlineLog>
        <BnH></BnH> 
        <RiskExposure></RiskExposure>
      </div>
    </div>
  )
}

export default App
