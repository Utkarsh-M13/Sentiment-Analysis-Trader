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
    <div className="w-full min-h-screen flex 2xl:gap-8 gap-6 flex-col justify-center items-center 2xl:pt-8 pt-6">
      <Topbar></Topbar>
      <div className="2xl:h-100 h-75 pl-7 pr-7 flex 2xl:gap-8 gap-5">
        <EquityCurve></EquityCurve>
        <div className="2xl:w-56 2xl:h-100 w-40 h-75 flex flex-col 2xl:gap-8 gap-5">
          <TotalEquity></TotalEquity>
          <SharpeRatio></SharpeRatio>
        </div>
        <TradeLog></TradeLog>
      </div>
      <div className="2xl:h-100 h-75 pl-7 pr-7 flex 2xl:gap-8 gap-5">
        <HeadlineLog></HeadlineLog>
        <BnH></BnH> 
        <RiskExposure></RiskExposure>
      </div>
    </div>
  )
}

export default App
